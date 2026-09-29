"""Conservative, auditable quality checks for a mixed coding RL dataset.

The AWS CodeFu article specifies quality checks, but does not publish their
full implementation. These checks implement the documented categories without
changing the requested difficulty ranges, original prompts, or test semantics.
"""

import ast
import hashlib
import html
import json
import math
import random
import re
import unicodedata
from collections import defaultdict


def canonical_statement(text):
    """Normalize formatting only; preserve mathematical operators and numbers."""
    text = html.unescape(unicodedata.normalize("NFKC", text))
    text = re.sub(r"-{5}(Input|Output|Examples?|Note)-{5}", r"\1", text, flags=re.I)
    return " ".join(text.casefold().split())


def statement_digest(text):
    return hashlib.sha256(canonical_statement(text).encode()).hexdigest()


def numeric_bound(text):
    """Read small numeric constraint expressions without executing dataset text."""
    text = text.replace("^", "**").replace("{", "").replace("}", "")
    tree = ast.parse(text.strip(), mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Pow)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 18 or abs(left) > 10**12:
                    raise ValueError("Unreasonably large constraint expression")
                value = left**right
            elif isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            else:
                value = left * right
        else:
            raise ValueError("Not a numeric constraint expression")
        if not math.isfinite(value) or abs(value) > 10**30:
            raise ValueError("Unreasonably large constraint value")
        return value

    return visit(tree)


def contradictory_bounds(text):
    text = html.unescape(unicodedata.normalize("NFKC", text)).replace("−", "-")
    text = text.replace(r"\leq", "<=").replace(r"\le", "<=").replace("≤", "<=")
    text = text.replace("$", "").replace(r"\times", "*").replace(r"\cdot", "*")
    text = text.replace("·", "*").replace("⋅", "*").replace("×", "*")
    # Parse whole numeric bounds, including 1 000, 1,000 and 1\,000. Never
    # compare a truncated prefix of an upper bound (e.g. 1 from 1 000).
    text = re.sub(r"(?<=\d)(?:[ \t]|,|'|\\,)(?=\d{3}(?!\d))", "", text)
    atom = r"\d+(?:\.\d+)?(?:\s*\^\s*\{?[+-]?\d+\}?)?"
    bound = rf"[+-]?\s*{atom}(?:\s*[+*-]\s*[+-]?\s*{atom})*"
    pattern = rf"(?<![\w.])({bound})\s*(<=|<)\s*([A-Za-z][\w_{{}}]*)\s*(<=|<)\s*({bound})(?![\w.])"
    bad = []
    for match in re.finditer(pattern, text):
        before, after = text[:match.start()].rstrip(), text[match.end():].lstrip()
        if before and before[-1] in "+-*/^\\{}":
            continue
        if after and (after[0].isdigit() or after[0] in "+-*/^\\{}_"):
            continue
        try:
            low, high = numeric_bound(match[1]), numeric_bound(match[5])
        except (SyntaxError, ValueError, OverflowError):
            continue
        if low > high or (low == high and "<" in (match[2], match[4])):
            bad.append(match[0])
    return bad


def statement_issues(text, *, standard_input=True):
    """Return exclusion reasons, not an assertion of semantic validity."""
    if not isinstance(text, str) or not text.strip():
        return ["missing_statement"]
    issues = []
    if len(text.strip()) < 100 or len(re.findall(r"[A-Za-z]{2,}", text)) < 15:
        issues.append("too_short_or_placeholder_statement")
    if "\ufffd" in text or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        issues.append("malformed_text_encoding")
    if re.search(r"(?is)^\s*(?:404\b|access denied|page not found|enable javascript)", text):
        issues.append("placeholder_statement")
    if re.search(r"(?i)\bthis is an? (?:interactive|output[- ]only) (?:problem|task)\b", text):
        issues.append("unsupported_interactive_or_output_only_task")
    if re.search(r"(?i)<(?:img|image)\b|!\[[^\]]*\]\(https?://", text):
        issues.append("external_image_in_statement")
    if contradictory_bounds(text):
        issues.append("contradictory_literal_constraint_bounds")
    if not re.search(r"(?i)constraints?|constrains|\\leq?\b|≤|<=|at most|at least|not exceed|(?:no|not) more than|between .{1,60} and|(?:range|length|from).{0,30}\d.{0,20} to \d", text):
        issues.append("no_explicit_constraint_evidence")
    if standard_input:
        if not re.search(r"\binput\b", text, re.I) or not re.search(r"\boutput\b", text, re.I):
            issues.append("missing_input_or_output_description")
        patterns = [
            r"if there (?:are|is|exist|exists) (?:multiple|several|many|more than one) (?:correct |possible |valid )?(?:answers?|solutions?).{0,160}?(?:print|output) any",
            r"(?:print|output) any (?:one |such |valid |correct |possible )?(?:answer|solution|permutation|arrangement|configuration)",
            r"(?:any|each) (?:valid|correct) (?:answer|solution).{0,40}(?:accepted|considered correct)",
            r"any of them.{0,40}(?:accepted|considered correct)",
            r"(?:multiple|several|many) (?:optimal |possible |valid |correct )?(?:answers?|solutions?).{0,100}?(?:find|choose|take) any\b",
            r"(?:print|output)[^.\n]{0,100}\bin any order\b",
        ]
        arbitrary_output = False
        for match in re.finditer(r"\b(?:print|output) any\b(?!\s+(?:additional|extra|debug|whitespace|case)\b)", text, re.I):
            prefix = re.split(r"[.\n]", text[max(0, match.start() - 100):match.start()])[-1]
            if not re.search(r"(?i)\b(?:do not|don't|must not|should not|not need to)\b", prefix):
                arbitrary_output = True
        if arbitrary_output or any(re.search(pattern, text, re.I | re.S) for pattern in patterns):
            issues.append("multiple_valid_outputs_without_special_judge")
    return sorted(set(issues))


def clean_stdio_tests(tests):
    """Check paired text cases and remove only byte-identical duplicate pairs."""
    if not isinstance(tests, dict):
        return None, ["missing_tests"], 0
    inputs, outputs = tests.get("inputs"), tests.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs:
        return None, ["missing_tests"], 0
    if len(inputs) != len(outputs):
        return None, ["unpaired_tests"], 0
    pairs, seen, by_input = [], set(), {}
    duplicate_count = 0
    for data, expected in zip(inputs, outputs):
        if not isinstance(data, str) or not isinstance(expected, str):
            return None, ["non_text_test_case"], 0
        if "\x00" in data or "\x00" in expected or "\ufffd" in data or "\ufffd" in expected:
            return None, ["malformed_test_encoding"], 0
        # Different expected outputs for exactly the same input are quarantined;
        # no guessed answer or constraint repair is made.
        output_key = "\n".join(line.strip().casefold() for line in expected.strip().splitlines())
        if data in by_input and by_input[data] != output_key:
            return None, ["conflicting_expected_outputs"], 0
        by_input[data] = output_key
        pair = (data, expected)
        if pair in seen:
            duplicate_count += 1
        else:
            seen.add(pair)
            pairs.append(pair)
    return {"inputs": [a for a, _ in pairs], "outputs": [b for _, b in pairs]}, [], duplicate_count


def leetcode_test_issues(row):
    required = ("task_id", "query", "prompt", "entry_point", "test", "completion")
    if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
        return ["missing_leetcode_checker_field"]
    try:
        tree = ast.parse(row["test"])
        ast.parse(row["prompt"] + "\n" + row["completion"] + "\n" + row["test"])
    except SyntaxError:
        return ["invalid_leetcode_test_or_reference_syntax"]
    if not any(isinstance(node, ast.FunctionDef) and node.name == "check" for node in ast.walk(tree)):
        return ["missing_leetcode_check_function"]
    if not any(isinstance(node, ast.Assert) for node in ast.walk(tree)):
        return ["missing_leetcode_assertions"]
    return []


def duplicate_groups(records):
    """Transitive groups from explicit source identities and conservative text hashes."""
    parents = list(range(len(records)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, record in enumerate(records):
        keys = set(record["identity_keys"]) | {"statement:" + record["statement_sha256"]}
        for key in keys:
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(records[index])
    return list(groups.values())


def balanced_fill(pools, count, seed=42):
    """Sample deterministically in round-robin order across available platforms."""
    rng = random.Random(seed)
    pools = {k: list(v) for k, v in sorted(pools.items())}
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    while len(selected) < count:
        progressed = False
        for key in pools:
            if pools[key] and len(selected) < count:
                selected.append(pools[key].pop())
                progressed = True
        if not progressed:
            raise ValueError("Insufficient distinct clean filler problems")
    return selected


def stratified_sample(pools, count, seed=42):
    """Proportional strata with largest-remainder quotas, without replacement."""
    sizes = {key: len(pool) for key, pool in sorted(pools.items())}
    total = sum(sizes.values())
    if count < 0 or count > total:
        raise ValueError("Sample size exceeds eligible population or is negative")
    if not count:
        return []
    quotas = {key: count * size // total for key, size in sizes.items()}
    remainder_order = sorted(sizes, key=lambda key: (-(count * sizes[key] % total), key))
    for key in remainder_order[:count - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(seed)
    selected = []
    for key in sizes:
        selected.extend(rng.sample(list(pools[key]), quotas[key]))
    return selected


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
