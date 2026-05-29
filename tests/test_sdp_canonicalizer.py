import numpy as np
import pytest

from fomtools.canonicalizers.sdp import SDP
from fomtools.verifier import Verifier


class TestAddPsdConeIndices:
    """Tests for SDP.add_psd_cone_indices set containment handling."""

    def test_new_set_contained_in_existing_not_added(self):
        """New set that is a subset of existing set should not be added."""
        sdp = SDP()
        sdp.psd_cone_indices = [{0, 1, 2, 3, 4}]

        sdp.add_psd_cone_indices([(1, 3)])  # produces {0, 1, 2}

        assert len(sdp.psd_cone_indices) == 1
        assert sdp.psd_cone_indices[0] == {0, 1, 2, 3, 4}

    def test_new_set_contains_existing_removes_subsets(self):
        """New superset should remove existing strict subsets."""
        sdp = SDP()
        sdp.psd_cone_indices = [{0, 1, 2}, {0, 3, 4}]

        sdp.add_psd_cone_indices([(1, 5)])  # produces {0, 1, 2, 3, 4}

        assert len(sdp.psd_cone_indices) == 1
        assert sdp.psd_cone_indices[0] == {0, 1, 2, 3, 4}

    def test_disjoint_sets_added_normally(self):
        """Disjoint sets should both be kept."""
        sdp = SDP()
        sdp.psd_cone_indices = [{0, 1, 2}]

        sdp.add_psd_cone_indices([(5, 7)])  # produces {0, 5, 6}

        assert len(sdp.psd_cone_indices) == 2
        assert {0, 1, 2} in sdp.psd_cone_indices
        assert {0, 5, 6} in sdp.psd_cone_indices

    def test_equal_sets_not_duplicated(self):
        """Adding an equal set should not create a duplicate."""
        sdp = SDP()
        sdp.psd_cone_indices = [{0, 1, 2, 3}]

        sdp.add_psd_cone_indices([(1, 4)])  # produces {0, 1, 2, 3}

        assert len(sdp.psd_cone_indices) == 1
        assert sdp.psd_cone_indices[0] == {0, 1, 2, 3}

    def test_partial_overlap_both_kept(self):
        """Partially overlapping sets (neither is subset) should both be kept."""
        sdp = SDP()
        sdp.psd_cone_indices = [{0, 1, 2, 3}]

        sdp.add_psd_cone_indices([(2, 5)])  # produces {0, 2, 3, 4}

        assert len(sdp.psd_cone_indices) == 2
        assert {0, 1, 2, 3} in sdp.psd_cone_indices
        assert {0, 2, 3, 4} in sdp.psd_cone_indices

    def test_multiple_subsets_removed(self):
        """A new superset should remove all its strict subsets."""
        sdp = SDP()
        sdp.psd_cone_indices = [{0, 1}, {0, 2}, {0, 3}]

        sdp.add_psd_cone_indices([(1, 4)])  # produces {0, 1, 2, 3}

        assert len(sdp.psd_cone_indices) == 1
        assert sdp.psd_cone_indices[0] == {0, 1, 2, 3}

    def test_empty_initial_list(self):
        """Adding to empty list should work."""
        sdp = SDP()
        sdp.psd_cone_indices = []

        sdp.add_psd_cone_indices([(1, 3)])  # produces {0, 1, 2}

        assert len(sdp.psd_cone_indices) == 1
        assert sdp.psd_cone_indices[0] == {0, 1, 2}

    def test_index_zero_always_included(self):
        """Index 0 should always be added for homogenization."""
        sdp = SDP()
        sdp.psd_cone_indices = []

        sdp.add_psd_cone_indices([(5, 8)])  # produces {0, 5, 6, 7}

        assert 0 in sdp.psd_cone_indices[0]
        assert sdp.psd_cone_indices[0] == {0, 5, 6, 7}


# TODO: Add integration test with a full verification problem that triggers
# multiple PSD cones once the SDP canonicalizer supports arbitrary constraints.


class TestCanonicalizeAffine:
    """
    Tests for ``SDP.canonicalize_affine``. The Affine step is encoded as a
    linear equality routed through ``add_constraint`` →
    ``_add_equality_constraint_impl``, which emits one linear matrix and
    one quadratic strengthening matrix per output coordinate plus a PSD
    cone over the involved leaves.
    """

    @staticmethod
    def _make_verifier():
        return Verifier(method="sdp", solver="clarabel", obbt_handler=None)

    def test_canonicalize_does_not_raise(self):
        """Issuing an Affine step on SDP completes without NotImplementedError."""
        v = self._make_verifier()
        x = v.add_iterate(3, lb=-1.0, ub=1.0)
        # Pure affine: y = W @ x with identity lhs_mat
        W = np.eye(3) * 2.0
        y = v.step.implicit_linear(rhs=W @ x, lhs_mat=np.eye(3))
        assert y.ndim == 3

    def test_emits_six_aeq_matrices_for_three_output_coords(self):
        """
        Each output coordinate contributes one linear matrix and one quadratic
        strengthening matrix to ``Aeq_matrices``: 2 * ndim total.
        """
        v = self._make_verifier()
        x = v.add_iterate(3, lb=-1.0, ub=1.0)
        prior = len(v.canonicalizer.Aeq_matrices)

        v.step.implicit_linear(rhs=2.0 * x, lhs_mat=np.eye(3))

        assert len(v.canonicalizer.Aeq_matrices) - prior == 2 * 3

    def test_emits_aeq_matrices_for_two_output_coords(self):
        """Coordinate count drives the matrix count: 2D output → 4 matrices."""
        v = self._make_verifier()
        x = v.add_iterate(2, lb=-1.0, ub=1.0)
        prior = len(v.canonicalizer.Aeq_matrices)

        v.step.implicit_linear(rhs=2.0 * x, lhs_mat=np.eye(2))

        assert len(v.canonicalizer.Aeq_matrices) - prior == 2 * 2

    def test_psd_cone_includes_affine_leaves(self):
        """
        The PSD cone group registered for the Affine step contains the
        homogenization index 0 plus the involved leaf indices.
        """
        v = self._make_verifier()
        x = v.add_iterate(3, lb=-1.0, ub=1.0)
        y = v.step.implicit_linear(rhs=2.0 * x, lhs_mat=np.eye(3))

        # At least one PSD cone group covers index 0 + every leaf index in {x, y}
        x_idx = v.canonicalizer.var_idx_map[x]
        y_idx = v.canonicalizer.var_idx_map[y]
        wanted = {0, *range(x_idx[0], x_idx[1]), *range(y_idx[0], y_idx[1])}
        assert any(wanted <= cone for cone in v.canonicalizer.psd_cone_indices)

    def test_end_to_end_solve_with_affine_and_relu(self):
        """
        Small SDP problem: y = -x + 1, z = relu(y), max ||z||_2^2.
        With x ∈ [-1, 1]^3, y ∈ [0, 2]^3 and z = y (since y ≥ 0); the
        squared norm should approach 12 (= 4 * 3).
        """
        v = self._make_verifier()
        x = v.add_iterate(3, lb=-1.0, ub=1.0)
        y = v.step.implicit_linear(rhs=-x + np.ones(3), lhs_mat=np.eye(3))
        z = v.step.relu(rhs=y)
        v.set_two_norm_objective([z])

        residual = v.solve()

        assert residual == pytest.approx(12.0, abs=1e-4)
