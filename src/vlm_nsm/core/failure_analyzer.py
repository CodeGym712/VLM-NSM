KNOWN_ALIGNMENT_ERRORS = {
    "invalid_object",
    "invalid_predicate",
    "type_error",
    "incomplete_goal",
}


def classify_failure(parse_result=None, alignment_result=None, plan_result=None, execution_result=None, goal_success=None):
    if parse_result is not None and not parse_result.success:
        return "json_parse_error" if parse_result.error == "json_parse_error" else "unknown_error"
    if alignment_result is not None and not alignment_result.success:
        return alignment_result.error if alignment_result.error in KNOWN_ALIGNMENT_ERRORS else "unknown_error"
    if plan_result is not None and not plan_result.success:
        return "planner_fail"
    if execution_result is not None and not execution_result.success:
        return "execution_fail"
    if goal_success is False:
        return "goal_not_satisfied"
    return "none"
