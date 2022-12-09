from inter_net.network import *

def make_model(rank, args, model_name=None):
    module = __import__('inter_net')
    class_ = getattr(module, args.model) if model_name is None else getattr(module, model_name)
    print(class_.__name__)
    model = class_(rank, args)
    return model
