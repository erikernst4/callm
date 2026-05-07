from functools import partial
from .constants import METRICS


def get_metric_from_id(id_name):
    if id_name in METRICS:
        metric_dict = METRICS[id_name]
        cls = metric_dict["cls"]
        metric_dict["obj"] = cls()
        return metric_dict
    elif "conf_n-ecuas" in id_name:
        n = int(id_name.split("_n=")[-1])
        return {
            "full_name": f"Confidence n={n}-ECUAS",
            "function": partial(METRICS["conf_n-ecuas"]["function"], n=n),
            "obj": METRICS["conf_n-ecuas"]["cls"](n=n),
            "higher_is_better": METRICS["conf_n-ecuas"]["higher_is_better"],
            "display": f"ECUAS$_{n}$",
        }
    elif "conf_gamma-ecuas" in id_name:
        gamma = float(id_name.split("_gamma=")[-1])
        return {
            "full_name": f"Confidence γ={gamma}-ECUAS",
            "function": partial(METRICS["conf_gamma-ecuas"]["function"], gamma=gamma),
            "obj": METRICS["conf_gamma-ecuas"]["cls"](gamma=gamma),
            "higher_is_better": METRICS["conf_gamma-ecuas"]["higher_is_better"],
            "display": f"ECUAS$^*$ (γ={gamma})",
        }
    elif "conf_ece" in id_name:
        if "nbins=" not in id_name:
            nbins = 10  # default
        else:
            nbins = int(id_name.split("_nbins=")[-1])
        return {
            "full_name": f"Confidence ECE (nbins={nbins})",
            "function": partial(METRICS["conf_ece"]["function"], nbins=nbins),
            "obj": METRICS["conf_ece"]["cls"](n_bins=nbins),
            "higher_is_better": METRICS["conf_ece"]["higher_is_better"],
            "display": "ECE$^*$" if nbins == 10 else f"ECE(n={nbins})$^*$",
        }
    elif "cls_n-ecuas" in id_name:
        n = int(id_name.split("_n=")[-1])
        return {
            "full_name": f"Classification n={n}-ECUAS",
            "function": partial(METRICS["cls_n-ecuas"]["function"], n=n),
            "obj": METRICS["cls_n-ecuas"]["cls"](n=n),
            "higher_is_better": METRICS["cls_n-ecuas"]["higher_is_better"],
            "display": f"ECUAS (n={n})",
        }
    elif "cls_norm_n-ecuas" in id_name:
        n = int(id_name.split("_n=")[-1])
        return {
            "full_name": f"Classification Normalized n={n}-ECUAS",
            "function": partial(METRICS["cls_norm_n-ecuas"]["function"], n=n),
            "obj": METRICS["cls_norm_n-ecuas"]["cls"](n=n),
            "higher_is_better": METRICS["cls_norm_n-ecuas"]["higher_is_better"],
            "display": f"ECUAS (n={n})",
        }
    elif "cls_gamma-ecuas" in id_name:
        gamma = float(id_name.split("_gamma=")[-1])
        return {
            "full_name": f"Classification γ={gamma}-ECUAS",
            "function": partial(METRICS["cls_gamma-ecuas"]["function"], gamma=gamma),
            "obj": METRICS["cls_gamma-ecuas"]["cls"](gamma=gamma),
            "higher_is_better": METRICS["cls_gamma-ecuas"]["higher_is_better"],
            "display": f"ECUAS (γ={gamma})",
        }
    elif "cls_norm_gamma-ecuas" in id_name:
        gamma = float(id_name.split("_gamma=")[-1])
        return {
            "full_name": f"Classification Normalized γ={gamma}-ECUAS",
            "function": partial(
                METRICS["cls_norm_gamma-ecuas"]["function"], gamma=gamma
            ),
            "obj": METRICS["cls_norm_gamma-ecuas"]["cls"](gamma=gamma),
            "higher_is_better": METRICS["cls_norm_gamma-ecuas"]["higher_is_better"],
            "display": f"ECUAS (γ={gamma})",
        }
    elif "cls_ece" in id_name:
        if "nbins=" not in id_name:
            nbins = 10  # default
        else:
            nbins = int(id_name.split("_nbins=")[-1])
        return {
            "full_name": f"Classification ECE (nbins={nbins})",
            "function": partial(METRICS["cls_ece"]["function"], nbins=nbins),
            "obj": METRICS["cls_ece"]["cls"](n_bins=nbins),
            "higher_is_better": METRICS["cls_ece"]["higher_is_better"],
            "display": "ECE" if nbins == 10 else f"ECE(n={nbins})",
        }
    else:
        raise ValueError(f"Metric not found: {id_name}")
