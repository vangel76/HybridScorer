import unittest

from lib.utils import normalize_prompt_text, parse_promptmatch_weighted_prompt
from lib.helpers import (
    sanitize_export_name,
    clamp_threshold,
    expand_slider_bounds,
    threshold_for_percentile,
    normalize_generated_prompt,
)
from lib.state_helpers import can_reuse_proxy_map
from lib.config import METHOD_PROMPTMATCH, METHOD_SIMILARITY


class NormalizePromptTextTests(unittest.TestCase):
    def test_collapses_whitespace(self):
        self.assertEqual(normalize_prompt_text("a  b   c"), "a b c")

    def test_strips_leading_trailing_commas(self):
        self.assertEqual(normalize_prompt_text(", hello ,"), "hello")

    def test_fixes_space_before_punctuation(self):
        self.assertEqual(normalize_prompt_text("hello , world"), "hello, world")

    def test_fixes_space_inside_brackets(self):
        self.assertEqual(normalize_prompt_text("( foo )"), "(foo)")

    def test_empty_returns_empty(self):
        self.assertEqual(normalize_prompt_text(""), "")
        self.assertEqual(normalize_prompt_text(None), "")

    def test_none_safe(self):
        self.assertEqual(normalize_prompt_text(None), "")


class ParsePromptmatchWeightedPromptTests(unittest.TestCase):
    def _parse(self, prompt):
        _rendered, fragments, segments = parse_promptmatch_weighted_prompt(prompt)
        return fragments, segments

    def test_plain_text_no_fragments(self):
        fragments, segments = self._parse("a dog on a hill")
        self.assertEqual(fragments, [])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["kind"], "text")

    def test_single_weighted_fragment(self):
        fragments, segments = self._parse("(sunset:1.5) sky")
        self.assertEqual(len(fragments), 1)
        self.assertAlmostEqual(fragments[0]["weight"], 1.5)
        self.assertEqual(fragments[0]["text"], "sunset")

    def test_multiple_weighted_fragments(self):
        fragments, _ = self._parse("(fog:0.8) (rain:1.2) dark alley")
        self.assertEqual(len(fragments), 2)
        texts = {f["text"] for f in fragments}
        self.assertIn("fog", texts)
        self.assertIn("rain", texts)

    def test_zero_weight_skipped(self):
        fragments, _ = self._parse("(ghost:0.0) visible")
        self.assertEqual(fragments, [])

    def test_empty_prompt(self):
        fragments, segments = self._parse("")
        self.assertEqual(fragments, [])


class SanitizeExportNameTests(unittest.TestCase):
    def test_replaces_bad_chars(self):
        self.assertEqual(sanitize_export_name("my/output:name"), "my-output-name")

    def test_strips_leading_trailing_dots_dashes(self):
        self.assertEqual(sanitize_export_name("--hello--"), "hello")

    def test_preserves_alphanumeric_and_dot_dash(self):
        self.assertEqual(sanitize_export_name("export-v1.2"), "export-v1.2")

    def test_empty_returns_empty(self):
        self.assertEqual(sanitize_export_name(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(sanitize_export_name(None), "")


class ClampThresholdTests(unittest.TestCase):
    def test_clamps_below(self):
        self.assertEqual(clamp_threshold(-1.0, 0.0, 1.0), 0.0)

    def test_clamps_above(self):
        self.assertEqual(clamp_threshold(2.0, 0.0, 1.0), 1.0)

    def test_within_range(self):
        self.assertEqual(clamp_threshold(0.5, 0.0, 1.0), 0.5)

    def test_rounds_to_3_decimal(self):
        self.assertEqual(clamp_threshold(0.12345, 0.0, 1.0), 0.123)


class ExpandSliderBoundsTests(unittest.TestCase):
    def test_expands_below_lo(self):
        lo, hi = expand_slider_bounds(0.1, 0.9, -0.5)
        self.assertLessEqual(lo, -0.5)

    def test_expands_above_hi(self):
        lo, hi = expand_slider_bounds(0.1, 0.9, 1.5)
        self.assertGreaterEqual(hi, 1.5)

    def test_no_expansion_when_within(self):
        lo, hi = expand_slider_bounds(0.0, 1.0, 0.5)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 1.0)

    def test_none_values_ignored(self):
        lo, hi = expand_slider_bounds(0.1, 0.9, None)
        self.assertAlmostEqual(lo, 0.1, places=2)
        self.assertAlmostEqual(hi, 0.9, places=2)

    def test_floor_ceil_prevents_out_of_bounds(self):
        lo, hi = expand_slider_bounds(0.5555, 0.6666)
        self.assertLessEqual(lo, 0.5555)
        self.assertGreaterEqual(hi, 0.6666)


class ThresholdForPercentileTests(unittest.TestCase):
    def _scores(self, vals):
        return {str(i): {"pos": v, "failed": False} for i, v in enumerate(vals)}

    def test_zero_percentile_returns_above_max(self):
        scores = self._scores([0.1, 0.5, 0.9])
        result = threshold_for_percentile(METHOD_PROMPTMATCH, scores, 0)
        self.assertGreater(result, 0.9)

    def test_100_percentile_returns_below_min(self):
        scores = self._scores([0.1, 0.5, 0.9])
        result = threshold_for_percentile(METHOD_PROMPTMATCH, scores, 100)
        self.assertLess(result, 0.1)

    def test_50_percentile_is_median_ish(self):
        scores = self._scores([0.2, 0.4, 0.6, 0.8])
        result = threshold_for_percentile(METHOD_PROMPTMATCH, scores, 50)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_empty_scores_returns_zero(self):
        result = threshold_for_percentile(METHOD_PROMPTMATCH, {}, 50)
        self.assertEqual(result, 0.0)


class NormalizeGeneratedPromptTests(unittest.TestCase):
    def test_strips_known_prefix(self):
        result = normalize_generated_prompt("This image shows a cat on a roof")
        self.assertNotIn("This image shows", result)
        self.assertIn("cat", result)

    def test_empty_returns_empty(self):
        self.assertEqual(normalize_generated_prompt(""), "")
        self.assertEqual(normalize_generated_prompt(None), "")

    def test_converts_sentences_to_comma_list(self):
        result = normalize_generated_prompt("a cat. a dog. a bird.")
        self.assertNotIn(". ", result)

    def test_keep_prose_preserves_sentences(self):
        result = normalize_generated_prompt("a cat. a dog.", keep_prose=True)
        self.assertIn(".", result)


class CanReuseProxyMapTests(unittest.TestCase):
    def _state(self, proxy_map, proxy_signature):
        return {"proxy_map": proxy_map, "proxy_signature": proxy_signature}

    def test_reuse_when_all_paths_present(self):
        paths = ["/img/a.png", "/img/b.png"]
        state = self._state({p: f"/tmp/{i}.jpg" for i, p in enumerate(paths)}, "sig1")
        self.assertTrue(can_reuse_proxy_map(state, paths, "sig1"))

    def test_no_reuse_on_wrong_signature(self):
        paths = ["/img/a.png"]
        state = self._state({"/img/a.png": "/tmp/0.jpg"}, "sig1")
        self.assertFalse(can_reuse_proxy_map(state, paths, "sig2"))

    def test_no_reuse_when_path_missing(self):
        state = self._state({"/img/a.png": "/tmp/0.jpg"}, "sig1")
        self.assertFalse(can_reuse_proxy_map(state, ["/img/a.png", "/img/b.png"], "sig1"))

    def test_no_reuse_when_empty_proxy_map(self):
        state = self._state({}, "sig1")
        self.assertFalse(can_reuse_proxy_map(state, ["/img/a.png"], "sig1"))

    def test_no_reuse_when_none_proxy_map(self):
        state = self._state(None, "sig1")
        self.assertFalse(can_reuse_proxy_map(state, ["/img/a.png"], "sig1"))


if __name__ == "__main__":
    unittest.main()
