def produce_name(member, variant):
    name = []
    if "l2ae" in member:
        return "L2-AE"
    if "AURORA_smaller" in member:
        return "AE-SMALL"
    if "AURORA" in member:
        return "AE"
    if "best" in member:
        return "RAED-LINPENALTY"
    if "sample" in member:
        return "RAED-KLPENALTY"
    if "largerbeta0" in member:
        return "RAED-LARGE"
    if "beta0" in member:
        return "RAED"
    if "standard" in member:
        return "RAED"
    if "largerbeta1" in member:
        return "RAED-LARGE-LINPENALTY"
    if "manualBD" in member:
        return "PREDEFINED-BD"
    return "-".join(name)