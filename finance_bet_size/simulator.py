import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


class Betting:

    def __init__(self, starting_sum: float, bet_fraction: float, p: float):
        self.p = 1-p
        self.starting_sum = starting_sum
        self.curr_val = [starting_sum]
        self.bet_fraction = bet_fraction

    def run(self, iters: int):
        results = [self.starting_sum]
        for i in range(iters):
            curr = results[-1]
            bet_size = curr * self.bet_fraction
            if curr <= 1:
                N = iters - len(results) + 1
                results.extend([0]*N)
                break
            results.append(curr + self.bet(bet_size))
        return results

    def bet(self, bet_size):
        p = np.random.rand()
        if p >= self.p:
            return bet_size
        else:
            return -1.0*bet_size


def plot(df: pd.DataFrame, output_loc: str):
    print('\n{star}\n{mod}\n{star}'.format(star=''.join(['*']*10), mod='Saving plot ({output_loc})...'))
    plt.figure(figsize=(12,9))
    with sns.axes_style("white"):
        g = sns.lineplot(
            x='Day',
            y='Portfolio_Value',
            hue='Betting_fraction',
            data=df
        )
        g.set_ylim(1e2, df.Portfolio_Value.max())
        g.set(yscale="log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(output_loc,
                bbox_inches='tight',
                transparent=True,
                pad_inches=0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--days', '-d', dest='days', type=int, default=364, help='Number of days to run the simulation for.')
    parser.add_argument('--betting-fractions', '-b', nargs='*', dest='betting_fractions', default=[0.15, 0.2, 0.25], help='Fractions of portfolio to bet.')
    parser.add_argument('--iterations', '-i', dest='iters', type=int, default=100, help='Number of iterations to run for each simulation.')
    parser.add_argument('--starting-sum', '-s', dest='starting_sum', type=float, default=50000.0, help='Starting sum in portfolio.')
    parser.add_argument('--win-probability', '-p', dest='probability_win', type=float, default=0.6, help='Winning probability for each bet.')
    parser.add_argument('--output', '-o', dest='output_loc', default='output.png', help='Plot output filename.')
    args = parser.parse_args()

    days = args.days
    bf = args.betting_fractions
    Q = dict.fromkeys(bf, [])
    iters = args.iters
    rs = [None]*iters
    p = args.probability_win
    S = args.starting_sum
    output_loc = args.output_loc
    print('\n{star}\n{mod}\n{star}'.format(star=''.join(['*']*10), mod=f'Running simulations of {", ".join(map(str, bf))} for {days} days ({iters} iteratiosn each)\n\t-- starting sum {S}\n\t-- win probability {p}'))
    for i in tqdm(range(iters)):
        for q in Q.keys():
            betting = Betting(S, q, p)
            v = betting.run(days)
            Q[q] = v

        _df = pd.DataFrame(Q)
        _df['Day'] = range(days+1)

        df_melt = _df.melt(
            value_name='Portfolio_Value',
            value_vars=_df.columns[:-1],
            var_name='Betting_fraction',
            id_vars='Day'
        )

        df_melt['Betting_fraction'] = df_melt['Betting_fraction'].apply(lambda f: str(int(f*100)) + '%')
        rs[i] = df_melt
    df = pd.concat(rs)
    plot(df, output_loc)
