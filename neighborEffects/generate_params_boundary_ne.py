from generate_params_boundary import *
from abmdiffuse_ne import Diffuse_ne


class Gen_para_ne(Gen_para):
    def __init__(self, alpha, g=nx.gnm_random_graph(10000, 30000), 
            p_cont=(0.001, 0.02), q_cont=(0.08, 0.1), delta=(0.0005, 0.01)):
        self.p_cont = p_cont
        self.q_cont = q_cont
        self.d_p, self.d_q = delta
        self.alpha = alpha
        self.g = g

    def add_data(self, p, q):
        diff = Diffuse_ne(p, q, alpha=self.alpha, g=self.g)
        x = np.mean(diff.repete_diffuse(), axis=0)
        max_idx = np.argmax(x)
        s = x[:(max_idx + 2)]
        para_range = [[1e-6, 0.1], [1e-5, 0.8],
                      [s, 4*self.g.number_of_nodes()]]
        bassest = BassEstimate(s, para_range)
        bassest.t_n = 1000
        res = bassest.optima_search(c_n=200, threshold=10e-6)
        return res[1:3]  # P, Q


def func(p, q, alpha):
    diff = Diffuse_ne(p, q, alpha=alpha)
    x = np.mean(diff.repete_diffuse(), axis=0)
    return np.concatenate(([p, q], x))


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.abmDiffusion
    prj = db.neighEffects
    alpha_cont = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 1.3, 1.5, 2]
    bound_dict = {}
    g = nx.gnm_random_graph(10000, 30000)
    for j, alpha in enumerate(alpha_cont):
        t1 = time.clock()
        print(j + 1, alpha_cont[j])
        p_cont = (0.0003, 0.02)
        q_cont = (0.076*3.0/(j + 4), 0.12*3.0/(j + 4))  # 小心设置
        delta = (0.0001, 0.001)
        ger_samp = Gen_para_ne(alpha, p_cont=p_cont, q_cont=q_cont, delta=delta)
        bound = ger_samp.identify_range()
        prj.insert_one({"_id": alpha, "para_boundary": bound})
        print(f'  time: {time.clock() - t1:.2f}s')
