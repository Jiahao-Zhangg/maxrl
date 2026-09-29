"""Lossless test-format conversion for the pinned LiveCodeBench Python grader.

This module never executes source tests or reference solutions. All conversions
are structural, and unsupported assertions fail closed instead of disappearing.
"""

import ast
import json
import math
import re
from collections import deque


LCB_REVISION = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
LCB_TESTING_SHA256 = "b7cb6a8a69807bb868150a61742e25d7bb5328bbe01d514471b6ec43c9fa9ed2"
ADAPTER_METHOD = "_maxrl_lcb_adapter_v1"


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def literal(node):
    """Only source literals and the dataset's explicit infinity constant."""
    if isinstance(node, ast.Name) and node.id == "inf":
        return math.inf
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Name) and node.operand.id == "inf":
            return -math.inf
    value = ast.literal_eval(node)
    dumps(value)
    return value


def tree_value(values):
    """Serialize exactly the tree built by LeetCodeDataset's tree_node helper."""
    if not isinstance(values, list):
        raise ValueError("Tree input must be a level-order list")
    if not values:
        return []
    nodes = [[values[0], None, None]]
    pending, index = deque([0]), 1
    while pending:
        parent = pending.popleft()
        for side in (1, 2):
            if index < len(values) and values[index] is not None:
                nodes[parent][side] = len(nodes)
                pending.append(len(nodes))
                nodes.append([values[index], None, None])
            index += 1
    output, pending = [], deque([0])
    while pending:
        index = pending.popleft()
        if index is None:
            output.append(None)
        else:
            value, left, right = nodes[index]
            output.append([value])
            pending.extend((left, right))
    while output and output[-1] is None:
        output.pop()
    return output


def source_value(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in {"tree_node", "list_node"} or len(node.args) != 1 or node.keywords:
            raise ValueError("Unsupported source test helper")
        value = literal(node.args[0])
        if not isinstance(value, list):
            raise ValueError("Node helper requires a list")
        return value, "tree" if node.func.id == "tree_node" else "linked_list"
    return literal(node), "json"


def leetcode_tests(row):
    """Convert every assertion, preserving its order and input argument order."""
    match = re.fullmatch(r"Solution\(\)\.([A-Za-z_][A-Za-z_0-9]*)", row["entry_point"])
    if not match:
        raise ValueError("Only an explicit Solution method is supported")
    name = match.group(1)
    # Dataset starter_code ends at the empty method body.
    starter = ast.parse(row["starter_code"].rstrip() + "\n        pass\n")
    methods = [node for node in ast.walk(starter) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(methods) != 1:
        raise ValueError("Ambiguous starter signature")
    signature = methods[0].args
    if signature.vararg or signature.kwarg or signature.kwonlyargs or signature.posonlyargs:
        raise ValueError("Unsupported starter signature")
    names = [arg.arg for arg in signature.args]
    if not names or names.pop(0) != "self":
        raise ValueError("Expected a Solution instance method")
    tree = ast.parse(row["test"])
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected one check function")
    check = tree.body[0]
    if check.name != "check" or [a.arg for a in check.args.args] != ["candidate"]:
        raise ValueError("Unexpected check signature")
    if not check.body or any(not isinstance(node, ast.Assert) for node in check.body):
        raise ValueError("Every statement in check must be an assertion")
    inputs, outputs, input_types, output_types = [], [], [], []
    has_infinity = False
    for statement in check.body:
        test = statement.test
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            candidate, expected = test.left, test.comparators[0]
            output, output_type = source_value(expected)
            if output_type != "json":
                raise ValueError("Node comparison must use the original structural helper")
        elif isinstance(test, ast.Call) and isinstance(test.func, ast.Name):
            kind = {"is_same_tree": "tree", "is_same_list": "linked_list"}.get(test.func.id)
            if kind is None or len(test.args) != 2 or test.keywords:
                raise ValueError("Unsupported assertion predicate")
            candidate, expected = test.args
            output, output_type = source_value(expected)
            if output_type != kind:
                raise ValueError("Mismatched structural assertion")
            if kind == "tree":
                output = tree_value(output)
        else:
            raise ValueError("Unsupported assertion")
        if not (isinstance(candidate, ast.Call) and isinstance(candidate.func, ast.Name)
                and candidate.func.id == "candidate" and not candidate.args):
            raise ValueError("Expected a keyword-only candidate call")
        kwargs = {item.arg: item.value for item in candidate.keywords}
        if len(kwargs) != len(candidate.keywords) or set(kwargs) != set(names):
            raise ValueError("Test arguments differ from the starter signature")
        values, types = zip(*(source_value(kwargs[key]) for key in names))
        for value in values:
            dumps(value)
        inputs.append("\n".join(dumps(value) for value in values))
        input_types.append(list(types))
        if isinstance(output, float) and not math.isfinite(output):
            if math.isnan(output):
                raise ValueError("NaN equality cannot be a passing test")
            has_infinity = True
            output = {"__lcb_float__": "inf" if output > 0 else "-inf"}
        outputs.append(dumps(output))
        output_types.append(output_type)
    structural_outputs = set(output_types) - {"json"}
    if len(structural_outputs) > 1:
        raise ValueError("Inconsistent output types")
    output_type = next(iter(structural_outputs), "json")
    if structural_outputs:
        for index, kind in enumerate(output_types):
            if kind == "json":
                if json.loads(outputs[index]) is not None:
                    raise ValueError("Structural output mixed with a non-null JSON output")
                outputs[index] = "[]"
    converters = []
    for position in range(len(names)):
        kinds = {types[position] for types in input_types}
        structural = kinds - {"json"}
        if len(structural) > 1:
            raise ValueError("Inconsistent input conversion across tests")
        if structural and "json" in kinds:
            for index, types in enumerate(input_types):
                if types[position] == "json" and json.loads(inputs[index].split("\n")[position]) is not None:
                    raise ValueError("Node input mixed with non-null JSON input")
        converters.append(next(iter(structural), "json"))
    adapted = any(t != "json" for t in converters) or output_type != "json" or has_infinity
    return {
        "input_output": {"inputs": inputs, "outputs": outputs,
                         "fn_name": ADAPTER_METHOD if adapted else name},
        "code_prelude": row["prompt"],
        "adapter": {"method": name, "argument_names": names, "input_types": converters,
                    "output_type": output_type, "normalize_infinity": has_infinity} if adapted else None,
        "test_count": len(inputs),
    }


ADAPTER_SOURCE = '''
def _maxrl_lcb_adapter_v1(self, *values):
    values = list(values)
    for index, kind in enumerate(_maxrl_lcb_settings["input_types"]):
        if kind == "tree":
            values[index] = tree_node(values[index])
        elif kind == "linked_list":
            values[index] = list_node(values[index])
    result = getattr(self, _maxrl_lcb_settings["method"])(*values)
    kind = _maxrl_lcb_settings["output_type"]
    if kind == "linked_list":
        output, seen = [], set()
        while result is not None:
            if id(result) in seen:
                raise ValueError("Cyclic list returned")
            seen.add(id(result))
            output.append(result.val)
            result = result.next
        return output
    if kind == "tree":
        from collections import deque
        pending, output = deque([(result, frozenset())]), []
        while pending:
            node, ancestors = pending.popleft()
            if node is None:
                output.append(None)
            else:
                if id(node) in ancestors:
                    raise ValueError("Cyclic tree returned")
                ancestors = ancestors | {id(node)}
                output.append([node.val])
                pending.extend(((node.left, ancestors), (node.right, ancestors)))
        while output and output[-1] is None:
            output.pop()
        return output
    if _maxrl_lcb_settings["normalize_infinity"]:
        import math
        if isinstance(result, float) and math.isinf(result):
            return {"__lcb_float__": "inf" if result > 0 else "-inf"}
        if isinstance(result, dict) and "__lcb_float__" in result:
            raise TypeError("Expected a numeric return value")
    return result
Solution._maxrl_lcb_adapter_v1 = _maxrl_lcb_adapter_v1
'''


def prepare_code(code, truth):
    prefix = truth.get("code_prelude", "")
    combined = prefix + "\n" + code if prefix else code
    if truth.get("adapter"):
        combined += "\n_maxrl_lcb_settings = " + repr(truth["adapter"]) + "\n" + ADAPTER_SOURCE
    return combined


def convert_truth(old, leetcode_row=None):
    truth = {"grader": "livecodebench", "schema_version": 1, "revision": LCB_REVISION,
             "unit_test_timeout_seconds": 10, "check_eos": False, "score_after_thinking": True}
    if old["grader"] == "nemo_gym_code_gen":
        tests = old["unit_tests"]
        if tests.get("fn_name") is not None:
            raise ValueError("Expected stdin Nemotron tests")
        truth.update(input_output={"inputs": tests["inputs"], "outputs": tests["outputs"], "fn_name": None},
                     code_prelude="", adapter=None)
    elif old["grader"] == "leetcode_dataset":
        if leetcode_row is None or old["problem"] != {k: leetcode_row[k] for k in old["problem"]}:
            raise ValueError("LeetCode source does not match the exported problem")
        converted = leetcode_tests(leetcode_row)
        converted.pop("test_count")
        truth.update(converted)
    else:
        raise ValueError("Unsupported source grader")
    validate_truth(truth)
    return truth


def validate_truth(truth):
    if (truth.get("grader") != "livecodebench" or truth.get("revision") != LCB_REVISION
            or truth.get("schema_version") != 1):
        raise ValueError("Wrong grader, source revision, or schema version")
    if truth.get("check_eos") is not False or truth.get("score_after_thinking") is not True:
        raise ValueError("Expected after-thinking grading without an EOS requirement")
    timeout = truth.get("unit_test_timeout_seconds")
    if type(timeout) is not int or not 1 <= timeout <= 60:
        raise ValueError("Invalid per-test timeout")
    tests = truth["input_output"]
    if not tests["inputs"] or len(tests["inputs"]) != len(tests["outputs"]):
        raise ValueError("Empty or unpaired tests")
    if any(not isinstance(v, str) for key in ("inputs", "outputs") for v in tests[key]):
        raise ValueError("LCB inputs and outputs must be strings")
    name = tests.get("fn_name")
    if name is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("Invalid method name")
    if truth.get("adapter") and name != ADAPTER_METHOD:
        raise ValueError("Adapter entry point mismatch")
    return truth
