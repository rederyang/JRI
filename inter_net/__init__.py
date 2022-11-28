from inter_net.network import *

def make_model(rank, args):
    module = __import__('inter_net')
    class_ = getattr(module, args.model)
    print(class_.__name__)
    model = class_(rank, args)
    return model
