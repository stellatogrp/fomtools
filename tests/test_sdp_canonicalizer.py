from fomtools.canonicalizers.sdp import SDP


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
