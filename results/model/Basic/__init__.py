from .recon import ISTAPlus, Basicmodel

__all__ = ["ISTAPlus", "Basicmodel", "ReconkneeFactory"]
class ReconkneeFactory:
    @staticmethod
    def get_recon_model(model_name, **kwargs):
        if model_name == "ISTAPlus":
            return ISTAPlus(**kwargs)
        elif model_name == "Basicmodel":
            return Basicmodel(**kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")