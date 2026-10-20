"""Module 6 — structural characteristics of Pharo methods.

Features are descriptive (flags and counts). Complexity measurements are NOT
interpreted as task difficulty.
"""
from __future__ import annotations

from ..extraction.smalltalk_parser import iter_nodes

CONDITIONAL = {"ifTrue:", "ifFalse:", "ifTrue:ifFalse:", "ifFalse:ifTrue:", "and:", "or:", "ifNil:", "ifNotNil:",
               "ifNil:ifNotNil:", "ifNotNil:ifNil:", "ifEmpty:", "ifNotEmpty:", "ifEmpty:ifNotEmpty:",
               "ifNotEmpty:ifEmpty:", "caseOf:", "caseOf:otherwise:", "ifEmptyOrNil:", "ifAbsent:"}
LOOPS = {"whileTrue:", "whileFalse:", "whileTrue", "whileFalse", "to:do:", "to:by:do:", "timesRepeat:", "repeat",
         "whileNotNil:", "whileNil:"}
ITERATION = {"do:", "collect:", "select:", "reject:", "detect:", "detect:ifNone:", "inject:into:", "doWithIndex:",
             "withIndexDo:", "keysAndValuesDo:", "reverseDo:", "do:separatedBy:", "keysDo:", "valuesDo:",
             "associationsDo:", "allSatisfy:", "anySatisfy:", "count:", "sum:", "detectMax:", "detectMin:",
             "flatCollect:", "select:thenCollect:", "collect:thenSelect:", "withIndexCollect:", "collectWithIndex:",
             "with:do:", "with:collect:", "groupedBy:", "noneSatisfy:", "select:thenDo:", "reject:thenDo:", "pairsDo:"}
EXCEPTION_HANDLING = {"on:do:", "ensure:", "ifCurtailed:", "on:do:on:do:", "retry", "resume", "resume:", "pass",
                      "outer", "return:", "retryUsing:"}
EXCEPTION_SIGNAL = {"signal", "signal:", "error:", "signalOn:", "signalFor:", "signal:with:", "signalIn:"}
DYNAMIC_SENDS = {"perform:", "perform:with:", "perform:with:with:", "perform:with:with:with:",
                 "perform:with:with:with:with:", "perform:withArguments:", "perform:withArguments:inSuperclass:",
                 "perform:withEnoughArguments:", "sendTo:", "valueWithReceiver:arguments:"}
REFLECTION = {"respondsTo:", "isKindOf:", "isMemberOf:", "instVarNamed:", "instVarNamed:put:", "instVarAt:",
              "instVarAt:put:", "allInstVarNames", "instVarNames", "selectors", "methodDict", "includesSelector:",
              "canUnderstand:", "lookupSelector:", "compiledMethodAt:", "allSubclasses", "subclasses", "inheritsFrom:",
              "allInstances", "allSelectors", "slots", "allSlots", "classVariables", "instanceSide", "classSide",
              "sourceCodeAt:", "slotNamed:", "pragmas", "allSubInstances", "someInstance", "allReferences"}
METAPROGRAMMING = {"compile:", "compile:classified:", "compile:classified:notifying:", "compile:notifying:",
                   "addInstVarNamed:", "removeInstVarNamed:", "removeSelector:", "addSelector:withMethod:",
                   "addAndClassifySelector:withMethod:inProtocol:", "become:", "becomeForward:", "addSlot:", "removeSlot:",
                   "subclass:instanceVariableNames:classVariableNames:package:",
                   "subclass:instanceVariableNames:classVariableNames:category:",
                   "subclass:instanceVariableNames:classVariableNames:poolDictionaries:category:",
                   "subclass:slots:classVariables:package:", "subclass:slots:classVariables:category:", "newSubclass",
                   "classInstaller", "make:", "recompile", "recompile:", "evaluate:", "removeFromSystem", "rename:",
                   "addClassVarNamed:", "removeClassVarNamed:", "adoptInstance:", "primitiveChangeClassTo:"}
INSTANCE_CREATION = {"new", "new:", "basicNew", "basicNew:", "new:withAll:", "with:", "with:with:", "with:with:with:",
                     "withAll:", "newFrom:", "new:streamContents:", "on:", "fromString:", "readFrom:", "x:y:"}
BLOCK_EVALUATION = {"value", "value:", "value:value:", "value:value:value:", "value:value:value:value:",
                    "valueWithArguments:", "cull:", "cull:cull:", "cull:cull:cull:", "numArgs",
                    "valueWithPossibleArgs:"}
ABSTRACT = {"subclassResponsibility", "explicitRequirement", "shouldBeImplemented"}


def _top_statements(ast: dict | None) -> list[dict]:
    return ((ast or {}).get("body") or {}).get("statements", []) if ast else []


def compute_features(method: dict, class_info: dict | None = None) -> dict:
    """Return descriptive feature flags and counts for one parsed method record."""
    sends = method.get("message_sends") or []
    selectors = [s["selector"] for s in sends]
    sel_set = set(selectors)
    ast = method.get("ast")
    stmts = _top_statements(ast)
    var_reads = method.get("variables_read") or {}
    assigns = method.get("assignments") or []
    returns = method.get("returns") or []
    args = method.get("arguments") or []
    own = method.get("selector")
    inst_reads = var_reads.get("instance_variable", [])
    inst_writes = sorted({a["variable"] for a in assigns if a.get("scope") == "instance_variable"})
    class_var_writes = sorted({a["variable"] for a in assigns if a.get("scope") == "class_variable"})

    recursive = [s for s in sends if s["selector"] == own and s["receiver_kind"] == "self"]
    iterates = bool(sel_set & ITERATION)
    f = {
        "n_ast_nodes": method.get("n_ast_nodes", 0),
        "n_message_sends": len(sends),
        "n_distinct_selectors": len(sel_set),
        "n_blocks": len(method.get("blocks") or []),
        "max_nesting_depth": method.get("max_block_depth", 0),
        "n_temporaries": len(method.get("temporaries") or []) + len(method.get("block_temporaries") or []),
        "n_arguments": len(args),
        "n_referenced_globals": len({g["name"] for g in method.get("referenced_globals") or []}),
        "n_statements": len(stmts),
        "n_literals": len(method.get("literals") or []),
        "n_assignments": len(assigns),
        "n_returns": len(returns),
        "n_cascades": 0,
        "n_lines": (method.get("source") or "").count("\n") + 1,
        "has_conditional": bool(sel_set & CONDITIONAL),
        "has_loop": bool(sel_set & LOOPS),
        "has_collection_iteration": iterates,
        "has_exception_handler": bool(sel_set & EXCEPTION_HANDLING),
        "has_exception_signal": bool(sel_set & EXCEPTION_SIGNAL),
        "has_recursion": bool(recursive),
        "recursion_in_iteration": any(s["block_depth"] > 0 for s in recursive) and iterates,
        "has_nonlocal_return": any(r.get("in_block") for r in returns),
        "has_early_return": bool(method.get("has_early_return")),
        "has_pragmas": bool(method.get("pragmas")),
        "pragma_selectors": [p["selector"] for p in method.get("pragmas") or []],
        "has_super_send": any(s["receiver_kind"] == "super" for s in sends),
        "has_self_send": any(s["receiver_kind"] == "self" for s in sends),
        "has_dynamic_send": bool(sel_set & DYNAMIC_SENDS),
        "has_reflection": bool(sel_set & REFLECTION),
        "has_metaprogramming": bool(sel_set & METAPROGRAMMING) or any(s.startswith("subclass:") for s in sel_set),
        "has_thisContext": "thisContext" in (var_reads.get("pseudo") or []) or any(s["receiver_kind"] == "thisContext" for s in sends),
        "has_global_environment_access": any(s["receiver_name"] == "Smalltalk" and s["receiver_kind"] == "global" for s in sends),
        "has_instance_creation": any(
            s["selector"] in INSTANCE_CREATION and (
                s["receiver_kind"] in ("global", "class_variable")
                or (s["receiver_kind"] == "self" and method.get("side") == "class")
                or (s["receiver_kind"] == "message_result" and s["receiver_name"] == "class"))
            for s in sends),
        "reads_instance_variables": bool(inst_reads),
        "writes_instance_variables": bool(inst_writes),
        "writes_class_variables": bool(class_var_writes),
        "instance_variables_read": inst_reads,
        "instance_variables_written": inst_writes,
        "has_block_argument": any("block" in s.get("argument_types", []) for s in sends),
        "has_stored_block": any(a.get("value_type") == "block" for a in assigns),
        "has_block_evaluation": any(s["selector"] in BLOCK_EVALUATION and s["receiver_kind"] in
                                    ("argument", "temporary", "block_argument", "instance_variable") for s in sends),
        "has_block_argument_parameter": any(s["selector"] in BLOCK_EVALUATION and s["receiver_kind"] == "argument" for s in sends),
        "is_abstract": bool(sel_set & ABSTRACT),
        "is_class_side": method.get("side") == "class",
        "is_extension": bool(method.get("is_extension")),
        "in_trait": bool(method.get("in_trait")),
        "class_uses_traits": bool(class_info and (class_info.get("traits") or class_info.get("class_traits"))),
        "is_test": bool(method.get("is_test")),
        "is_empty": method.get("n_ast_nodes", 0) <= 2 and not stmts,
        "is_getter": False,
        "is_setter": False,
        "is_lazy_initializer": False,
        "has_cascade": False,
    }
    if ast:
        cascades = sum(1 for n, *_ in iter_nodes(ast) if n.get("type") == "Cascade")
        f["n_cascades"] = cascades
        f["has_cascade"] = cascades > 0
        if len(stmts) == 1 and stmts[0].get("type") == "Return":
            v = stmts[0].get("value") or {}
            if v.get("type") == "Variable" and v.get("name") in inst_reads and not args:
                f["is_getter"] = True
            if (v.get("type") == "Message" and v.get("selector") in ("ifNil:", "ifNil:ifNotNil:")
                    and (v.get("receiver") or {}).get("name") in inst_reads and inst_writes):
                f["is_lazy_initializer"] = True
        if len(args) == 1 and 1 <= len(stmts) <= 2:
            first = stmts[0]
            if (first.get("type") == "Assignment" and first.get("variable") in inst_writes
                    and (first.get("value") or {}).get("type") == "Variable"
                    and (first.get("value") or {}).get("name") == args[0]):
                f["is_setter"] = True
    return f


LANGUAGE_FEATURES = [
    # (feature id, flag, concept_type) — RQ3 Pharo language capabilities
    ("message_sends", lambda f: f["n_message_sends"] > 0, "general"),
    ("blocks_closures", lambda f: f["n_blocks"] > 0, "pharo_specific"),
    ("block_values_stored_or_evaluated", lambda f: f["has_stored_block"] or f["has_block_evaluation"], "pharo_specific"),
    ("nonlocal_return", lambda f: f["has_nonlocal_return"], "pharo_specific"),
    ("cascades", lambda f: f["has_cascade"], "pharo_specific"),
    ("class_side_methods", lambda f: f["is_class_side"], "pharo_specific"),
    ("extension_methods", lambda f: f["is_extension"], "pharo_specific"),
    ("traits", lambda f: f["in_trait"] or f["class_uses_traits"], "pharo_specific"),
    ("pragmas", lambda f: f["has_pragmas"], "pharo_specific"),
    ("reflection", lambda f: f["has_reflection"], "pharo_specific"),
    ("dynamic_message_sends", lambda f: f["has_dynamic_send"], "pharo_specific"),
    ("metaprogramming", lambda f: f["has_metaprogramming"], "pharo_specific"),
    ("thisContext", lambda f: f["has_thisContext"], "pharo_specific"),
    ("super_sends", lambda f: f["has_super_send"], "general"),
    ("exception_handling", lambda f: f["has_exception_handler"] or f["has_exception_signal"], "general"),
    ("recursion", lambda f: f["has_recursion"], "general"),
    ("conditionals_as_messages", lambda f: f["has_conditional"], "pharo_specific"),
    ("loops_as_messages", lambda f: f["has_loop"], "pharo_specific"),
    ("collection_protocol_iteration", lambda f: f["has_collection_iteration"], "general"),
    ("instance_creation", lambda f: f["has_instance_creation"], "general"),
    ("abstract_methods", lambda f: f["is_abstract"], "general"),
]


def language_features(f: dict) -> list[str]:
    return [name for name, pred, _ in LANGUAGE_FEATURES if pred(f)]
