class ConstraintType:
    max = "max"
    min = "min"
    datatype = "datatype"
    disallow_nan = "disallow_nan"
    is_categorical = "is_categorical"
    min_domain_mass = "min_domain_mass"
    min_count = "presence.min_count"
    min_fraction = "presence_min_fraction"
    append_value_to_domain = "string_domain.value"
    categorical_drift_threshold = "categorical_drift_threshold"
    numerical_drift_threshold = "numerical_drift_threshold"

    @classmethod
    def allowed_values(cls):
        result = []
        for key, value in cls.__dict__.items():
            if (
                    not key.startswith("__") and
                    not key.startswith("allowed_values")
            ):
                result.append(value)
        return result
