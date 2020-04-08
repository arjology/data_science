import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats.mstats import gmean
from scipy.stats import gstd
from tqdm import tqdm


class Betting:
    def __init__(self, starting_sum: float, bet_fraction: float):
        self.starting_sum = starting_sum
        self.curr_val = [starting_sum]
        self.bet_fraction = bet_fraction

    def run(self, days: int):
        results = [self.starting_sum]
        for i in range(days):
            curr = results[-1]
            bet_size = curr * self.bet_fraction
            if curr <= 1:
                N = days - len(results) + 1
                results.extend([1]*N)
                break
            results.append(curr + self.bet(bet_size))
        return results

    def bet(self, bet_size):
        p = np.random.normal(loc=0.6, scale=0.1)
        _p = np.random.normal(loc=0.6, scale=0.1)
        if p >= _p:
            return bet_size
        else:
            return -1.0 * bet_size


def plot(min_val: float, df: pd.DataFrame, output_loc: str):
    plt.figure(figsize=(12, 9))
    with sns.axes_style("white"):
        plt.figure(figsize=(12,9))
        g = sns.lineplot(
            x='Day', y='Portfolio_Value', hue='Betting_fraction',
            data=df
        )
        g.set_ylim(min_val, df['Portfolio_Value'].max())
        g.set(yscale="log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        plt.savefig(output_loc, bbox_inches="tight", transparent=False, pad_inches=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days",
        "-d",
        dest="days",
        type=int,
        default=364,
        help="Number of days to run the simulation for.",
    )
    parser.add_argument(
        "--betting-fractions",
        "-b",
        nargs="*",
        type=float,
        dest="betting_fractions",
        default=[0.15, 0.2, 0.25],
        help="Fractions of portfolio to bet.",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        dest="iters",
        type=int,
        default=100,
        help="Number of iterations to run for each simulation.",
    )
    parser.add_argument(
        "--starting-sum",
        "-s",
        dest="starting_sum",
        type=float,
        default=50000.0,
        help="Starting sum in portfolio.",
    )
    parser.add_argument(
        "--win-probability",
        "-p",
        dest="probability_win",
        type=float,
        default=0.6,
        help="Winning probability for each bet.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_loc",
        default="output.png",
        help="Plot output filename.",
    )
    args = parser.parse_args()

    days = args.days
    days_array = np.arange(0, days+1)
    iters = args.iters
    bf = args.betting_fractions
    p = args.probability_win
    S = args.starting_sum
    output_loc = args.output_loc
    print(
        "\n{star}\n{mod}\n{star}".format(
            star="".join(["*"] * 10),
            mod=f'''Running simulations of {", ".join(map(str, bf))}
            -- for {days} days ({iters} iteratiosn each)
            -- starting sum {S}
            -- win probability {p}'''
        )
    )
    for method in tqdm([gmean, np.median]):
        Q = dict.fromkeys(bf, np.zeros((iters, days+1)))
        for q in tqdm(Q.keys()):
            for i in range(iters):
                betting = Betting(S, q)
                v = betting.run(days)
                Q[q][i] = v
            _tmp = Q[q].copy()
            Q[q] = method(_tmp, axis=0)

        df = pd.DataFrame(Q)
        df['Day'] = days_array
        df_melt = df.melt(
            value_name='Portfolio_Value',
            value_vars=df.columns[:-1],
            var_name='Betting_fraction',
            id_vars='Day'
        )
        df_melt['Betting_fraction'] = df_melt['Betting_fraction'].apply(lambda g: str(g*100)+'%')
        plot(S*.1, df=df_melt, output_loc=method.__name__ + '.png')
