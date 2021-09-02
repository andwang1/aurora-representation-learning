def produce_name(member, variant):
    name = []
    if "l2ae" in member:
        return "L2-AE"
    if "AURORA" in member:
        return "AE"
    if "best" in member:
        return "RAED-LINPENALTY"
    if "sample" in member:
        return "BD-KLPENALTY"
    if "beta0" in member:
        return "RAED"
    if "manualBD" in member:
        return "PREDEFINED-BD"
    return "-".join(name)