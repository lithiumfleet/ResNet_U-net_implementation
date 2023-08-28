from torch import tensor
def get_dic(mode:str):
    raw_str = 'person, bird, cat, cow, dog, horse, sheep, aeroplane, bicycle, boat, bus, car, motorbike, train, bottle, chair, diningtable, pottedplant, sofa, tvmonitor'
    names = raw_str.split(', ')
    # print(names)
    vectors = []
    for i in range(20):
        vec = [0] * 20
        vec[i] = 1
        t = tensor(vec)
        vectors.append(t)
    # print(vectors)
    if mode == 'cls2vec':
        return dict(zip(names,vectors))
    elif mode == 'vec2cls':
        return dict(zip(vectors,names))
    elif mode == 'num2cls':
        return dict(zip(range(20),names))
    else:
        return dict(zip(names,range(20)))
    



Cls2Vec = get_dic('cls2vec')

Vec2Cls = get_dic('vec2cls')

Num2Cls =  get_dic('num2cls') # for multi-label

Cls2Num = get_dic('cls2num')