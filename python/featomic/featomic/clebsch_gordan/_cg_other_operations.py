"""
Module for computing the Clebsch-Gordan tensor product between two arbitrary
:py:class:`TensorMap` objects.
"""

from typing import Callable, List, Optional, Set, Tuple

import numpy as np

from . import _coefficients, _dispatch, _utils
from ._backend import (
    Array,
    Device,
    DType,
    Labels,
    LabelsEntry,
    TensorBlock,
    TensorMap,
    TorchModule,
    TorchScriptClass,
    torch_jit_is_scripting,
)


try:
    import torch
except ImportError:
    pass


from ._cg_product import ClebschGordanProduct

class ClebschGordanSelfProduct(ClebschGordanProduct):
    """
    A general class for computing the Clebsch-Gordan (CG) tensor product between two
    channels of a single :py:class:`TensorMap`.

    The constructor computes and stores the CG coefficients. The :py:meth:`compute`
    method computes the CG tensor product between two tensors.
    """

    def compute(
        self,
        tensor: TensorMap,
        o3_lambda_1_new_name: str,
        o3_lambda_2_new_name: str,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Computes the Clebsch-Gordan (CG) tensor product between two channels of
        ``tensor``.

        This function assumes the metadata of ``tensor`` has been modified to be
        compatible for the CG tensor product, according to the following rules:

        - ``tensor`` must have key dimensions ``"o3_lambda1"``, ``"o3_lambda2"``,
          ``"o3_sigma1"``, and ``"o3_sigma2"`` as these are used to determine the
          symmetry of output blocks upon combination;
        - ``o3_lambda_1_new_name`` and ``o3_lambda_2_new_name`` define the output key
          dimension names that store the ``"o3_lambda1"`` and ``"o3_lambda2"`` values;
        - ``tensor`` must have two component axes with key dimensions
          ``"o3_mu1"``, ``"o3_mu2"``.

        A :py:class:`Labels` object can be passed in ``selected_keys`` to select
        specific keys to compute. The full set of output keys are computed, then are
        filtered to match the key dimensions passed in ``selected_keys``.

        This parameter can be used to match the output tensor to a given target basis
        set definition, and/or enhance performance by limiting the combinations
        computed.

        For instance, passing just the ``"o3_lambda"`` key dimension with a range of
        values ``0, ..., max_angular`` can be used to perform angular selection,
        limiting the maximum angular momentum of the output blocks.

        Note that using ``selected_keys`` to perform any kind of selection will reduce
        the information content of the output tensor. This may be important if the
        returned tensor is used in further CG tensor products.

        :param tensor: The :py:class:`TensorMap`, containing data with double O(3)
            character.
        :param o3_lambda_1_new_name: :py:class:`str`, the name of the output key
            dimension that stores the ``"o3_lambda"`` values of the blocks combined from
            ``tensor_1``.
        :param o3_lambda_2_new_name: :py:class:`str`, the name of the output key
            dimension that stores the ``"o3_lambda"`` values of the blocks combined from
            ``tensor_2``.
        :param selected_keys: :py:class:`Labels`, the keys to compute on the final
            iteration. If ``None``, all keys are computed. Subsets of key dimensions can
            be passed to compute output blocks that match in these dimensions.

        :return: A :py:class:`TensorMap` containing the Clebsch-Gordan tensor product of
            ``tensor`` with itself.
        """

        return self._cg_tensor_product(
            tensor,
            o3_lambda_1_new_name=o3_lambda_1_new_name,
            o3_lambda_2_new_name=o3_lambda_2_new_name,
            selected_keys=selected_keys,
            compute_metadata=False,
        )

    def forward(
        self,
        tensor: TensorMap,
        o3_lambda_1_new_name: str,
        o3_lambda_2_new_name: str,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Calls the :py:meth:`ClebschGordanSelfProduct.compute` function.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(
            tensor,
            o3_lambda_1_new_name=o3_lambda_1_new_name,
            o3_lambda_2_new_name=o3_lambda_2_new_name,
            selected_keys=selected_keys,
        )

    def compute_metadata(
        self,
        tensor: TensorMap,
        o3_lambda_1_new_name: str,
        o3_lambda_2_new_name: str,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self._cg_tensor_product(
            tensor,
            o3_lambda_1_new_name=o3_lambda_1_new_name,
            o3_lambda_2_new_name=o3_lambda_2_new_name,
            selected_keys=selected_keys,
            compute_metadata=True,
        )

    def _cg_tensor_product(
        self,
        tensor: TensorMap,
        o3_lambda_1_new_name: str,
        o3_lambda_2_new_name: str,
        selected_keys: Optional[Labels],
        compute_metadata: bool,
    ) -> TensorMap:
        """
        Computes the Clebsch-Gordan tensor product between ``tensor_1`` and
        itself.

        Executes the following steps:

            1. Checks the metadata of ``tensor`` to ensure that it is compatible for
               the Clebsch-Gordan tensor product. Ensures neither has gradients
               (not currently supported).

            2. Computes the full product of the two keys, accounting for CG combination
               rules, and matching key dimensions that have the same name.

            3. Applies the key selection to remove keys that do not need computing. This
               includes both ``selected_keys`` and any user-provided ``keys_filter``.

            4. For each key pair to be combined (and corresponding output angular
               channel), computes the CG tensor product on the block values.

            5. Returns the resulting tensor in :py:class:`TensorMap` format.
        """
        # 1. Check the inputs
        self._check_inputs(tensor, selected_keys)

        # 2. Compute the full product of the two keys
        output_keys, combinations = _compute_output_keys(
            tensor.keys, o3_lambda_1_new_name, o3_lambda_2_new_name
        )

        # 3. a) Apply key selections
        if selected_keys is not None:
            selected_idx = output_keys.select(selected_keys)
            combinations = [combinations[i] for i in selected_idx]
            output_keys = Labels(
                names=output_keys.names, values=output_keys.values[selected_idx]
            )

        # 3. b) Apply key filter
        keys_to_keep = self._keys_filter(output_keys)
        output_keys = Labels(output_keys.names, output_keys.values[keys_to_keep])
        combinations = [combinations[i] for i in keys_to_keep]

        # Group keys such that all o3_lambda values produced by the same 2 blocks are
        # together. This allows the block combination and CG product to be separated and
        # thus not unnecessarily repeated.

        output_keys, combinations = _group_combinations_of_same_blocks(
            output_keys, combinations
        )

        # 4. Do the CG tensor product for each block combination
        output_blocks: List[TensorBlock] = []
        for (block_i, o3_lambdas) in combinations:
            output_blocks.extend(
                cg_tensor_product_blocks(
                    tensor.block(block_i),
                    o3_lambdas=o3_lambdas,
                    cg_coefficients=self._cg_coefficients,
                    cg_backend="metadata" if compute_metadata else self._cg_backend,
                )
            )

        # 5. Build and return the resulting TensorMap
        return TensorMap(output_keys, output_blocks)

    def _check_inputs(
        self,
        tensor: TensorMap,
        selected_keys: Optional[Labels] = None,
    ) -> None:
        """
        Checks the metadata of ``tensor_1`` and ``tensor_2`` to ensure that they are
        compatible for the Clebsch-Gordan tensor product.

        This checks that:

            - Both tensors have "o3_lambda" and "o3_sigma" key dimensions.
            - Both tensors have a single component axis with a single dimension named
              "o3_mu".
            - The sum of maximum "o3_lambda" values in each tensor is not greater than
              the maximum angular momentum ``self._max_angular`` used to calculate the
              CG coefficients in the constructor.
            - There is no intersection of property names between the two tensors.
        """
        # Check symmetry dimensions
        for name in ["o3_lambda1", "o3_lambda2", "o3_sigma1", "o3_sigma2"]:
            if name not in tensor.keys.names:
                raise ValueError(
                    f"input tensors must have a named key dimension `{name}`"
                )
        if not tensor.component_names == ["o3_mu1", "o3_mu2"]:
            raise ValueError(
                "input tensors must have a single component"
                " axis with a single named dimensions `['o3_mu1', 'o3_mu2']`"
            )

        # Check maximum angular momentum
        max_angular: int = int(
            _dispatch.max(tensor.keys.column("o3_lambda1"))
            + _dispatch.max(tensor.keys.column("o3_lambda2"))
        )
        if selected_keys is not None:
            # Update `max_angular` if angular selection is being performed via
            # `selected_keys`
            if "o3_lambda" in selected_keys.names:
                # Check max of 'o3_lambda' column is valid
                for lam in selected_keys.column("o3_lambda"):
                    if lam > self._max_angular:
                        raise ValueError(
                            "the maximum angular momentum value found in key dimension"
                            " `'o3_lambda'` in `selected_keys` exceeds"
                            f" `max_angular={self._max_angular}`"
                            " used to calculate the CG coefficients in the constructor."
                        )
                max_angular = min(
                    max_angular, int(_dispatch.max(selected_keys.column("o3_lambda")))
                )
            if "o3_sigma" in selected_keys.names:
                # Check 'o3_sigma' column is valid
                for o3_sigma in selected_keys.column("o3_sigma"):
                    if int(o3_sigma) not in [-1, 1]:
                        raise ValueError(
                            "key dimension `'o3_sigma'` in `selected_keys` must only"
                            " contain values of -1 or 1."
                        )
        if max_angular > int(self._max_angular):
            raise ValueError(
                f"the maximum angular momentum `{max_angular}` required by the"
                " input tensors (and perhaps lowered by angular selection via"
                f" `selected_keys`) exceeds the `max_angular={self._max_angular}`"
                " used to calculate the CG coefficients in the constructor."
            )

        # Check no gradients as not currently supported.
        contains_gradients = all(
            [len(list(block.gradients())) > 0 for _, block in tensor.items()]
        )
        if contains_gradients:
            raise NotImplementedError(
                "Clebsch Gordan combinations with gradients not yet implemented."
                " Use `metatensor.remove_gradients` to remove gradients from the input"
                " tensors."
            )


def _keys_filter_noop(keys: Labels):
    """A ``key_filter`` for ``ClebschGordanProduct`` that does nothing"""
    return [i for i in range(len(keys))]


def _compute_output_keys(
    keys: Labels,
    o3_lambda_1_new_name: str,
    o3_lambda_2_new_name: str,
) -> Tuple[Labels, List[int]]:
    """
    Computes the output keys from the full CG tensor product of two channels of ``keys``.

    The output keys are the full product of the keys in ``keys``, where
    Clebsch-Gordan combination rules are used to determine the new angular order and
    parity of output blocks. Key dimensions "o3_lambda1", "o3_lambda2",
    "o3_sigma1", and "o3_sigma2" must therefore be present.

    ``o3_lambda_1_new_name`` and ``o3_lambda_2_new_name`` are the names of output key
    dimensions which store the "o3_lambda1" and "o3_lambda2" values respectively.

    For keys that are not named "o3_lambdaX" or "o3_sigmaX", we only take correlations
    between blocks that have the same values for all matching keys.

    Returned is a :py:class:`Labels` object for the output keys, and a list of the pairs
    of indices in ``keys_1`` and ``keys_2`` that combine to form each key in the output
    :py:class:`Labels`.

    :param keys: :py:class:`Labels`, the keys of the tensor.
    :param o3_lambda_1_new_name: str, the name of the output key dimension storing the
        "o3_lambda1" values.
    :param o3_lambda_2_new_name: str, the name of the output key dimension storing the
        "o3_lambda2" values.
    """

    # Define the names of the 'standard' (required) CG key names
    cg_names = ["o3_lambda1", "o3_sigma1", "o3_lambda2", "o3_sigma2"]
    for name in cg_names:
        assert name in keys.names, f"`keys` must contain dimension {name}"

    # Define the names and column idxs of the 'match' keys, if any
    match_names: List[str] = []
    for name in keys.names:
        if name in cg_names:
            continue
        else:
            match_names.append(name)
    match_idxs: List[int] = [keys.names.index(name) for name in match_names]

    # Build the new names: this enforces a certain order in the output keys
    output_names: List[str] = (
        ["o3_lambda", "o3_sigma"]
        + match_names
        + [o3_lambda_1_new_name]
        + [o3_lambda_2_new_name]
    )

    # Build the new keys by combination
    output_key_values: List[List[int]] = []
    combinations: List[int] = []
    for key_i in range(len(keys)):
            # Get the keys
            key = keys.entry(key_i)


            # Unpack standard key values
            o3_lambda_1, o3_sigma_1 = int(key["o3_lambda1"]), int(key["o3_sigma1"])
            o3_lambda_2, o3_sigma_2 = int(key["o3_lambda2"]), int(key["o3_sigma2"])

            # Create all non-zero angular channel combinations
            for o3_lambda in range(
                abs(o3_lambda_1 - o3_lambda_2), abs(o3_lambda_1 + o3_lambda_2) + 1
            ):
                # Calculate new sigma
                o3_sigma = int(
                    o3_sigma_1
                    * o3_sigma_2
                    * (-1) ** (o3_lambda_1 + o3_lambda_2 + o3_lambda)
                )

                # Build the new keys values
                new_vals: List[int] = (
                    [o3_lambda, o3_sigma]  # standard CG dims
                    + _dispatch.to_int_list(key.values[match_idxs])  # match dims
                    + [o3_lambda_1]  # lambda from block_1
                    + [o3_lambda_2]  # lambda from block_2
                )
                output_key_values.append(new_vals)
                combinations.append(key_i)

    # Define new keys as the full product of keys_1 and keys_2
    output_keys = Labels(
        names=output_names,
        values=_dispatch.int_array_like(output_key_values, like=keys.values),
    )

    return output_keys, combinations

def _group_combinations_of_same_blocks(
    output_keys: Labels,
    combinations: List[int],
) -> Tuple[Labels, List[Tuple[int,List[int]]]]:
    """
    Re-order output keys in such a way that we have all the different ``o3_lambda``
    created by the same pair of block consecutive in ``output_keys``.

    This function simultaneously returns the new ``output_keys``, and a list of
    combinations in the same order as the keys. Iterating over the returned combinations
    and then over the ``o3_lambdas`` for each combination will produce blocks in the
    same order as the output keys.
    """
    # block => list of index in the output keys
    # this is emulating a dict by separating keys & values since TorchScript does not
    # support Tuple[int, int] as Dict keys. `groups` is the dict keys and `groups_keys`
    # the dict values.
    groups: List[int] = []
    groups_keys: List[List[int]] = []
    for key_i, block in enumerate(combinations):
        group_id = -1
        for existing_id, existing in enumerate(groups):
            if existing == block:
                group_id = existing_id
                break

        if group_id == -1:
            group_id = len(groups)
            groups.append(block)
            keys_idx: List[int] = []
            groups_keys.append(keys_idx)

        groups_keys[group_id].append(key_i)

    all_o3_lambdas = output_keys.column("o3_lambda")

    keys_values: List[List[int]] = []
    combinations: List[Tuple[int,List[int]]] = []
    for block, keys_idx in zip(groups, groups_keys):
        o3_lambdas: List[int] = []
        for key_i in keys_idx:
            keys_values.append(_dispatch.to_int_list(output_keys.values[key_i]))
            o3_lambdas.append(int(all_o3_lambdas[key_i]))

        combinations.append((block, o3_lambdas))

    output_keys = Labels(
        output_keys.names,
        _dispatch.int_array_like(keys_values, like=output_keys.values),
    )

    return output_keys, combinations

def cg_tensor_product_blocks(
    block: TensorBlock,
    o3_lambdas: List[int],
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> List[TensorBlock]:
    """
    For a given pair of TensorBlocks and desired angular channels, combines the values
    arrays and returns a new TensorBlock.

    If the samples of the two blocks are different, we assume that one of the sample
    names are a subset of the other sample names. We then match sample values for
    this subset, and reshape the block with the smallest sample names to have the
    same number of samples as the other one.
    """

    properties = block.properties

    # Do the CG combination - single center so no shape pre-processing required
    combined_values = _coefficients.cg_tensor_selfproduct(
        block.values, o3_lambdas, cg_coefficients, cg_backend
    )

    # Create the TensorBlocks
    results: List[TensorBlock] = []
    for values, o3_lambda in zip(combined_values, o3_lambdas):
        block = TensorBlock(
            values=values,
            samples=block.samples,
            components=[
                Labels(
                    names=["o3_mu"],
                    values=_dispatch.int_range_like(
                        min_val=-o3_lambda,
                        max_val=o3_lambda + 1,
                        like=block.samples.values,
                    ).reshape(-1, 1),
                ),
            ],
            properties=properties,
        )
        results.append(block)

    return results





class ClebschGordanSplit(ClebschGordanProduct):
    """
    A general class for computing the Clebsch-Gordan (CG) tensor product between two
    channels of a single :py:class:`TensorMap`.

    The constructor computes and stores the CG coefficients. The :py:meth:`compute`
    method computes the CG tensor product between two tensors.
    """

    def compute(
        self,
        tensor: TensorMap,
        o3_lambda_new_name: str,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Computes the Clebsch-Gordan (CG) tensor splitting of an angular channel
        within a list of ``tensors``.

        This function assumes the metadata of the ``tensor`` has been modified
        to be compatible for the CG tensor product, according to the following rules:

        - the ``tensor`` must have key dimensions ``"o3_lambda"``, ``"o3_sigma"``
          as these are used to determine the symmetry of output blocks upon splitting;
        - for every requested ouput key ``o3_lambda1,o3_lambda2,o3_sigma1,o3_sigma2``,
          there must be block with values of ``o3_lambda`` between
          ``|o3_lambda1-o3_lambda2|`` and ``o3_lambda1+o3_lambda2``, and the
          correct values of ``"o3_sigma"``.
        - ``tensor`` must have one component axis with key dimension ``"o3_mu"``.

        A :py:class:`Labels` object can be passed in ``selected_keys`` to select
        specific keys to compute. The full set of output keys are computed, then are
        filtered to match the key dimensions passed in ``selected_keys``.

        This parameter can be used to match the output tensor to a given target basis
        set definition, and/or enhance performance by limiting the combinations
        computed.

        :param tensor: The :py:class:`TensorMap`, containing data with O(3) character.
        :param selected_keys: :py:class:`Labels`, the keys to compute on the final
            iteration. If ``None``, all keys are computed. Subsets of key dimensions can
            be passed to compute output blocks that match in these dimensions.

        :return: A :py:class:`TensorMap` containing the Clebsch-Gordan tensor product of
            ``tensor`` with itself.
        """

        return self._cg_tensor_product(
            tensor,
            o3_lambda_new_name,
            selected_keys=selected_keys,
            compute_metadata=False,
        )

    def forward(
        self,
        tensor: TensorMap,
        o3_lambda_new_name: str,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Calls the :py:meth:`ClebschGordanSplit.compute` function.

        This is intended for :py:class:`torch.nn.Module` compatibility, and should be
        ignored in pure Python mode.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self.compute(
            tensor,
            o3_lambda_new_name,
            selected_keys=selected_keys,
        )

    def compute_metadata(
        self,
        tensor: TensorMap,
        o3_lambda_new_name: str,
        selected_keys: Optional[Labels] = None,
    ) -> TensorMap:
        """
        Returns the metadata-only :py:class:`TensorMap` that would be output by the
        function :py:meth:`compute` for the same calculator under the same settings,
        without performing the actual Clebsch-Gordan tensor products.

        See :py:meth:`compute` for a full description of the parameters.
        """
        return self._cg_tensor_product(
            tensor,
            o3_lambda_new_name,
            selected_keys=selected_keys,
            compute_metadata=True,
        )

    def _cg_tensor_product(
        self,
        tensor: TensorMap,
        o3_lambda_new_name: str,
        selected_keys: Optional[Labels],
        compute_metadata: bool,
    ) -> TensorMap:
        """
        Computes the Clebsch-Gordan tensor product between ``tensor_1`` and
        itself.

        Executes the following steps:

            1. Checks the metadata of ``tensor`` to ensure that it is compatible for
               the Clebsch-Gordan tensor product. Ensures neither has gradients
               (not currently supported).

            2. Computes the full product of the two keys, accounting for CG combination
               rules, and matching key dimensions that have the same name.

            3. Applies the key selection to remove keys that do not need computing. This
               includes both ``selected_keys`` and any user-provided ``keys_filter``.

            4. For each key pair to be combined (and corresponding output angular
               channel), computes the CG tensor product on the block values.

            5. Returns the resulting tensor in :py:class:`TensorMap` format.
        """
        # 1. Check the inputs
        self._check_inputs(tensor, selected_keys)

        # 2. Compute the full product of the two keys
        output_keys, combinations = _compute_split_output_keys(tensor.keys)

        # 3. a) Apply key selections
        if selected_keys is not None:
            selected_idx = output_keys.select(selected_keys)
            combinations = [combinations[i] for i in selected_idx]
            output_keys = Labels(
                names=output_keys.names, values=output_keys.values[selected_idx]
            )

        # 3. b) Apply key filter
        keys_to_keep = self._keys_filter(output_keys)
        output_keys = Labels(output_keys.names, output_keys.values[keys_to_keep])
        combinations = [combinations[i] for i in keys_to_keep]

        # Group keys such that all o3_lambda values produced by the same 2 blocks are
        # together. This allows the block combination and CG product to be separated and
        # thus not unnecessarily repeated.

        # output_keys, combinations = _group_combinations_of_same_blocks(
        #     output_keys, combinations
        # )

        # 4. Do the CG tensor product for each block combination

        output_blocks: List[TensorBlock] = cg_tensor_split_blocks(
            tensor,
            combinations,
            o3_lambda_new_name,
            output_keys,
            cg_coefficients=self._cg_coefficients,
            cg_backend="metadata" if compute_metadata else self._cg_backend,
        )

        # 5. Build and return the resulting TensorMap
        return TensorMap(output_keys, output_blocks)

    def _check_inputs(
        self,
        tensor: TensorMap,
        selected_keys: Optional[Labels] = None,
    ) -> None:
        """
        Checks the metadata of ``tensor_1`` and ``tensor_2`` to ensure that they are
        compatible for the Clebsch-Gordan tensor product.

        This checks that:

            - Both tensors have "o3_lambda" and "o3_sigma" key dimensions.
            - Both tensors have a single component axis with a single dimension named
              "o3_mu".
            - The sum of maximum "o3_lambda" values in each tensor is not greater than
              the maximum angular momentum ``self._max_angular`` used to calculate the
              CG coefficients in the constructor.
            - There is no intersection of property names between the two tensors.
        """
        # Check symmetry dimensions
        for name in ["o3_lambda1", "o3_lambda2", "o3_sigma1", "o3_sigma2"]:
            if name in tensor.keys.names:
                raise ValueError(
                    f"input tensors must not have a named key dimension `{name}`"
                )
        for name in ["o3_lambda", "o3_sigma"]:
            if name not in tensor.keys.names:
                raise ValueError(
                    f"input tensors must have a named key dimension `{name}`"
                )
        if not tensor.component_names == ["o3_mu"]:
            raise ValueError(
                "input tensors must have a single component"
                " axis with a single named dimension `['o3_mu']`"
            )

        # Check maximum angular momentum
        max_angular: int = int(
            _dispatch.max(tensor.keys.column("o3_lambda"))
        )
        if selected_keys is not None:
            # Update `max_angular` if angular selection is being performed via
            # `selected_keys`
            lam_max = 0
            if "o3_lambda1" in selected_keys.names:
                # Check max of 'o3_lambda' column is valid
                for lam in selected_keys.column("o3_lambda1"):
                    if lam > lam_max:
                        lam_max = lam
            if "o3_lambda2" in selected_keys.names:
                # Check max of 'o3_lambda' column is valid
                for lam in selected_keys.column("o3_lambda2"):
                    if lam > lam_max:
                        lam_max = lam
            if lam_max > self._max_angular:
                raise ValueError(
                    "the maximum angular momentum value found in key dimension"
                    " `'o3_lambda'` in `selected_keys` exceeds"
                    f" `max_angular={self._max_angular}`"
                    " used to calculate the CG coefficients in the constructor."
                )
            max_angular = min(
                max_angular, int(_dispatch.max(
                    selected_keys.column("o3_lambda1")+selected_keys.column("o3_lambda2")
                ))
            )
            if "o3_sigma1" in selected_keys.names:
                # Check 'o3_sigma' column is valid
                for o3_sigma in selected_keys.column("o3_sigma1"):
                    if int(o3_sigma) not in [-1, 1]:
                        raise ValueError(
                            "key dimension `'o3_sigma'` in `selected_keys` must only"
                            " contain values of -1 or 1."
                        )
            if "o3_sigma2" in selected_keys.names:
                # Check 'o3_sigma' column is valid
                for o3_sigma in selected_keys.column("o3_sigma2"):
                    if int(o3_sigma) not in [-1, 1]:
                        raise ValueError(
                            "key dimension `'o3_sigma'` in `selected_keys` must only"
                            " contain values of -1 or 1."
                        )
        if max_angular > int(self._max_angular):
            raise ValueError(
                f"the maximum angular momentum `{max_angular}` required by the"
                " input tensors (and perhaps lowered by angular selection via"
                f" `selected_keys`) exceeds the `max_angular={self._max_angular}`"
                " used to calculate the CG coefficients in the constructor."
            )

        # Check no gradients as not currently supported.
        contains_gradients = all(
            [len(list(block.gradients())) > 0 for _, block in tensor.items()]
        )
        if contains_gradients:
            raise NotImplementedError(
                "Clebsch Gordan combinations with gradients not yet implemented."
                " Use `metatensor.remove_gradients` to remove gradients from the input"
                " tensors."
            )


def _compute_split_output_keys(
    keys: Labels,
) -> Tuple[Labels, List[List[int]]]:
    """
    Computes the output keys from the full CG tensor product, splitting a channels of ``keys``.

    The output keys are the full product of the keys in ``keys``, where
    Clebsch-Gordan combination rules are used to determine the new angular orders and
    parity of output blocks. Key dimensions "o3_lambda", and "o3_sigma",
    must therefore be present.

    For keys dimentions that are not named as such, we only take correlations
    between blocks that have the same values for all matching keys.

    Returned is a :py:class:`Labels` object for the output keys, and a list of the lists
    of indices in ``keys`` that combine to form each key in the output
    :py:class:`Labels`.

    :param keys: :py:class:`Labels`, the keys of the tensor.
    """

    # Define the names of the 'standard' (required) CG key names
    cg_out_names = ["o3_lambda1", "o3_sigma1", "o3_lambda2", "o3_sigma2"]
    cg_in_names = ["o3_lambda", "o3_sigma"]
    for name in cg_in_names:
        assert name in keys.names, f"`keys` must contain dimension {name}"
    for name in cg_out_names:
        assert name not in keys.names, f"`keys` must not contain dimension {name}"

    # Define the names and column idxs of the 'match' keys, if any
    match_names: List[str] = []
    for name in keys.names:
        if name in cg_in_names:
            continue
        else:
            match_names.append(name)
    match_idxs: List[int] = [keys.names.index(name) for name in match_names]

    # Build the new names: this enforces a certain order in the output keys
    output_names: List[str] = (
        cg_out_names
        + match_names
    )

    ls_lists: List[Tuple[
        List[Tuple[int,LabelsEntry]],
        List[Tuple[int,LabelsEntry]],
    ]] = []
    all_matches: List[LabelsEntry] = list()

    for key_i in range(len(keys)):
        key = keys.entry(key_i)
        o3_lambda, o3_sigma = int(key["o3_lambda"]), int(key["o3_sigma"])
        while len(ls_lists) <= o3_lambda:
            ls_lists.append(([],[]))

        new_key = LabelsEntry(match_names, key.values[match_idxs])
        #if new_key not in ls_lists[o3_lambda][int(o3_sigma!=1)]:
        #    ls_lists[o3_lambda][int(o3_sigma!=1)].append(new_key)
        ls_lists[o3_lambda][int(o3_sigma!=1)].append((key_i,new_key))
        if new_key not in all_matches:
            all_matches.append(new_key)

    match_lists: List[List[List[Optional[int]]]] = [
        [[None, None] for _ in range(len(ls_lists))]
        for _ in range(len(all_matches))
    ]

    for o3_lambda in range(len(ls_lists)):
        for sigma_i in (0,1):
            for key_i, match_key in ls_lists[o3_lambda][sigma_i]:
                match_lists[all_matches.index(match_key)][o3_lambda][sigma_i] = key_i

    # Build the new keys by combination
    output_key_values: List[Array] = []
    combinations: List[List[int]] = []

    for match_key_i in range(len(all_matches)):
        match_key = all_matches[match_key_i]
        available_ls = match_lists[match_key_i]

        highest_l = len(available_ls)-1
        while not (available_ls[highest_l][0] or available_ls[highest_l][1]):
            highest_l -=1
            if highest_l == -1:
                break
        if highest_l == -1:
            continue


        output_key_values_buffer: List[List[int]] = []

        for l_first in range(highest_l+1):
            highest_l_second: int = highest_l - l_first
            for l_second in range(highest_l_second+1):
                has_same_sigma = True
                same_sigma_keys: List[int] = []
                has_different_sigma = True
                different_sigma_keys: List[int] = []

                for o3_lambda in range(abs(l_first-l_second), l_first+l_second+1):
                    o3_sigma_i_same = (l_first+l_second+o3_lambda)%2

                    same_sigma_i = available_ls[o3_lambda][o3_sigma_i_same]
                    different_sigma_i = available_ls[o3_lambda][1-o3_sigma_i_same]
                    if same_sigma_i is None:
                        has_same_sigma = False
                    else:
                        same_sigma_keys.append(same_sigma_i)
                    if different_sigma_i is None:
                        has_different_sigma = False
                    else:
                        different_sigma_keys.append(different_sigma_i)

                if has_same_sigma:
                    combinations.append(same_sigma_keys)
                    combinations.append(same_sigma_keys)
                    output_key_values_buffer.append([l_first,1,l_second,1])
                    output_key_values_buffer.append([l_first,-1,l_second,-1])
                if has_different_sigma:
                    combinations.append(different_sigma_keys)
                    combinations.append(different_sigma_keys)
                    output_key_values_buffer.append([l_first,1,l_second,-1])
                    output_key_values_buffer.append([l_first,-1,l_second,1])

        output_key_values += [
            _dispatch.concatenate([
                _dispatch.int_array_like(out_key, like=keys.values),
                match_key.values,
            ], axis=0)
            for out_key in output_key_values_buffer
        ]



    # Define new keys as the full product of keys_1 and keys_2
    output_keys = Labels(
        names=output_names,
        values=_dispatch.int_array_like(output_key_values, like=keys.values),
    )

    return output_keys, combinations


def cg_tensor_split_blocks(
    tensor: TensorMap,
    combinations: List[List[int]],
    o3_lambda_new_name: str,
    output_keys: Labels,
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> List[TensorBlock]:
    """
    For a given pair of TensorBlocks and desired angular channels, combines the values
    arrays and returns a new TensorBlock.

    If the samples of the two blocks are different, we assume that one of the sample
    names are a subset of the other sample names. We then match sample values for
    this subset, and reshape the block with the smallest sample names to have the
    same number of samples as the other one.
    """

    # ##### compute the right combination lookups for the lambda -> L1,L2 splitting
    inverse_combinations: List[List[Tuple[int,int]]] = [[] for _ in range(len(tensor))]
    for output_i, combination in enumerate(combinations):
        for input_i_pre, input_i in enumerate(combination):
            inverse_combinations[input_i].append((output_i, input_i_pre))

    # ##### then, some setup, and listing which properties are to be used per output block
    output_values: List[Optional[Array]] = [None] * len(output_keys)
    # full_property_lists: List[List[LabelsEntry]] = [[] for _ in range(len(output_keys))]
    # for output_i, combination in enumerate(combinations):
    #     props: Set[LabelsEntry] = set()
    #     for block_i in combination:
    #         block = tensor.block(block_i)
    #         props.update(tuple(val) for val in block.properties.values)
    #     full_property_lists[output_i] = sorted(props)

    full_property_lists: List[Array] = [None for _ in range(len(output_keys))]
    full_property_boundaries: List[List[Tuple[int,int]]] = [[] for _ in range(len(output_keys))]
    for output_i, combination in enumerate(combinations):
        props: List[LabelsEntry] = []
        bounds = full_property_boundaries[output_i] = [None] * len(combination)
        cur: int = 0
        for block_i_pre, block_i in enumerate(combination):
            block = tensor.block(block_i)
            prop_sz = block.properties.values.shape[0]
            bounds[block_i_pre] = (cur, cur+prop_sz)
            cur += prop_sz
            o3_lam = tensor.keys[block_i]["o3_lambda"]
            vals = _dispatch.concatenate([
                block.properties.values,
                _dispatch.int_array_like([[o3_lam]]*len(block.properties.values), block.properties.values)
            ], axis=1)
            props += [vals]
        full_property_lists[output_i] = _dispatch.concatenate(props, axis=0)

    ref_blocks: List[Optional[TensorBlock]] = [None] * len(output_keys)
    #print(full_property_lists)

    for input_i, (inv_combination, block) in enumerate(zip(inverse_combinations, tensor)):
        ll_combinations: List[Tuple[int,int]] = []
        out_buffer: List[Array] = [None] * len(inv_combination)
        for out_i_pre, (out_i, input_i_pre) in enumerate(inv_combination):
            out_key = output_keys[out_i]
            ref_blocks[out_i] = block
            o3_lambda1, o3_lambda2 = int(out_key[0]), int(out_key[2])
            ll_combinations.append((o3_lambda1, o3_lambda2))

            if output_values[out_i] is None:
                pre_buf: List[Array] = _coefficients.cg_tensor_split(
                    block.values,
                    ll_combinations,
                    cg_coefficients, "metadata",
                )
                output_values[out_i] = _dispatch.zeros_like(
                    block.values,
                    (block.shape[0], 2*o3_lambda1+1, 2*o3_lambda2+1, full_property_boundaries[out_i][-1][1]),
                )
                #output_values[out_i][:] += np.nan  # TODO degub

            out_val = output_values[out_i]
            these_bounds = full_property_boundaries[out_i][input_i_pre]
            out_buffer[out_i_pre] = out_val[
                :,
                :,:,
                these_bounds[0]:these_bounds[1]
            ]

        _coefficients.cg_tensor_split(
            block.values,
            ll_combinations,
            cg_coefficients, cg_backend,
            out_arrays = out_buffer,
        )

        # for out_i_pre, out_i in enumerate(inv_combination):
        #     # ##### with the main calculation done for a given input->output pair
        #     # we need to distribute the properties right.
        #     prop_list = full_property_lists[out_i]
        #     prop_lut: List[int] = [0] * block.properties.values.shape[0]
        #     overall_prop_i = 0
        #     overall_prop = prop_list[overall_prop_i]
        #     for prop_i, prop in enumerate(block.properties.values):
        #         prop = tuple(prop)
        #         while overall_prop < prop:
        #             overall_prop_i +=1
        #             if overall_prop_i >= len(prop_list):
        #                 raise ValueError("internal issue, we didn't match properties in CG splitting")
        #             overall_prop = prop_list[overall_prop_i]
        #         if overall_prop == prop:
        #             prop_lut[prop_i] = overall_prop_i
        #         else:
        #             raise ValueError("internal issue, we can't match properties in CG splitting")


    # Create the TensorBlocks
    results: List[Optional[TensorBlock]] = [None] * len(output_keys)
    for out_i, (values, out_key, ref_block) in enumerate(zip(output_values, output_keys, ref_blocks)):
        o3_lambda1, o3_lambda2 = int(out_key[0]), int(out_key[2])

        mm_range1 = _dispatch.int_range_like(
            min_val=-o3_lambda1,
            max_val=o3_lambda1 + 1,
            like=ref_block.samples.values,
        )
        mm_range2 = _dispatch.int_range_like(
            min_val=-o3_lambda2,
            max_val=o3_lambda2 + 1,
            like=ref_block.samples.values,
        )
        #mm_range = _dispatch.concatenate([mm_range1[:,None,None], mm_range2[None,:,None]], axis=2)
        #mm_range = mm_range.reshape(-1,2)

        # if not np.isfinite(values).all():
        #     v2 = values.sum(axis=(0,1,2))
        #     bad_idx = np.nonzero(1- np.isfinite(v2))[0]
        #     print(bad_idx)
        #     for pre_i, prop_idx in enumerate(bad_idx):
        #         print(out_i,prop_idx, values[...,prop_idx], full_property_lists[out_i][prop_idx])
        #     raise ValueError('robin')

        block = TensorBlock(
            values=values,
            samples=ref_block.samples,
            components=[
                Labels(
                    names=["o3_mu1"],
                    values=mm_range1.reshape(-1,1),
                ),
                Labels(
                    names=["o3_mu2"],
                    values=mm_range2.reshape(-1,1),
                ),
            ],
            properties=Labels(
                ref_block.properties.names + [o3_lambda_new_name],
                _dispatch.int_array_like(full_property_lists[out_i], ref_block.properties.values),
            )
        )

        results[out_i] = block

    return results
