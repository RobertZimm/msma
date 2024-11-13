import numpy as np



def two_rel_machines(mu_1: float, mu_2: float, C: int, runtime: int):
    state_times = [0 for _ in range(C+3)]
    TH_i = [0 for _ in range(2)] # 
    t = 0
    current_state = 0

    while t < runtime:
        if current_state == 0:
            draw = np.random.exponential(1/mu_1)
            t += draw
            state_times[current_state] += draw
            TH_i[0] += 1
            current_state += 1
        
        elif current_state == C+2:
            draw = np.random.exponential(1/mu_2)
            t += draw
            state_times[current_state] += draw
            TH_i[0] += 1
            # TH_i[1] += 1
            current_state -= 1

        else:
            draw_mu_1 = np.random.exponential(1/mu_1)
            draw_mu_2 = np.random.exponential(1/mu_2)

            if draw_mu_1 < draw_mu_2:
                t += draw_mu_1
                state_times[current_state] += draw_mu_1

                if current_state <= C:
                    TH_i[0] += 1

                current_state += 1

            else:
                t += draw_mu_2
                state_times[current_state] += draw_mu_2
                
                # TH_i[1] += 1

                current_state -= 1

    TH_i[0] /= t
    TH_i[1] /= t
    # TH_i[1] = TH_i[0] * (mu_1/mu_2)

    for i in range(len(state_times)):
        state_times[i] /= t

    n_bar = sum([i * state_times[i] for i in range(len(state_times))])

    return n_bar, state_times, TH_i


def calc_pi(C: int, mu_1: float, mu_2: float):
    pi = np.zeros(C+3)
    pi[0] = (1-(mu_1/mu_2))/(1-(mu_1/mu_2)**(C+3))

    for i in range(1, C+3):
        pi[i] = (mu_1/mu_2)**i * pi[0]

    return pi


def calc_n_bar(pi: list):
    return sum([i * pi[i] for i in range(C+3)])


def calc_TH1(C: int, mu_1: float, mu_2: float, pi: list):
    return mu_1 * sum(pi[:C+1]) + mu_2 * pi[C+2]


def calc_TH2(C: int, mu_1: float, pi: list):
    return mu_1 * sum(pi[1:C+3])



if __name__ == "__main__":
    C = 2
    mu_1 = 0.3
    mu_2 = 0.6

    pi = calc_pi(C, mu_1, mu_2)
    n_bar = calc_n_bar(pi)
    TH_i = [calc_TH1(C, mu_1, mu_2, pi), calc_TH2(C, mu_1, pi)]

    n_bar_exp, pi_exp, TH_i_exp = two_rel_machines(mu_1, mu_2, C, 1e5)

    print(n_bar, n_bar_exp)
    print(pi, pi_exp)
    print(TH_i, TH_i_exp)
    print(calc_TH1(C, mu_1, mu_2, pi_exp), calc_TH2(C, mu_1, pi_exp))
