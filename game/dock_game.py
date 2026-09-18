import argparse
import random
import time
import matplotlib.pyplot as plt

class DockGame:
    def __init__(self, piles=(7,7,7)):
        self.piles=tuple(piles)

    def moves(self,state):
        result=[]

        for i,pile in enumerate(state):
            for take in range(1,min(3,pile)+1):
                s=list(state); s[i]-=take; result.append((i,take,tuple(s)))

        return result

    def terminal(self,state): return sum(state)==0

    def value(self,state): return -1 if self.terminal(state) else 0

class Search:
    def __init__(self,game): self.game=game; self.nodes=0

    def evaluate(self,state):
        if self.game.terminal(state): return -1
        nim=sum(state[i] for i in range(len(state)))
        return 1 if nim%4 else -1

    def minimax(self,state,depth,maximizing=True):
        self.nodes+=1
        if depth==0 or self.game.terminal(state): return self.evaluate(state),None
        best=(-float("inf"),None) if maximizing else (float("inf"),None)

        for move in self.game.moves(state):
            score,_=self.minimax(move[2],depth-1,not maximizing)
            if maximizing and score>best[0]: best=(score,move)
            if not maximizing and score<best[0]: best=(score,move)

        return best

    def alphabeta(self,state,depth,alpha=-float("inf"),beta=float("inf"),maximizing=True):
        self.nodes+=1
        if depth==0 or self.game.terminal(state): return self.evaluate(state),None
        moves=self.ordered_moves(state)

        if maximizing:
            value=-float("inf"); choice=None
            for move in moves:
                score,_=self.alphabeta(move[2],depth-1,alpha,beta,False)

                if score>value: value,choice=score,move
                alpha=max(alpha,value)
                if alpha>=beta: break

            return value,choice

        value=float("inf"); choice=None
        for move in moves:
            score,_=self.alphabeta(move[2],depth-1,alpha,beta,True)

            if score<value: value,choice=score,move
            beta=min(beta,value)
            if alpha>=beta: break

        return value,choice

    def ordered_moves(self,state):
        return sorted(self.game.moves(state),key=lambda m: sum(m[2]),reverse=True)

    def ids(self,state,max_depth):
        last=None; total=0; timings=[]

        for depth in range(1,max_depth+1):
            self.nodes=0; started=time.perf_counter(); value,move=self.alphabeta(state,depth); elapsed=time.perf_counter()-started
            total+=self.nodes; timings.append((depth,self.nodes,elapsed,value,move)); last=move

        self.nodes=total

        return last,timings

def chart(depths=range(2,7),start=(7,7,7),path="nodes_explored.png"):
    game=DockGame(start); mm=[]; ab=[]; ids=[]

    for d in depths:
        s=Search(game); s.minimax(start,d); mm.append(s.nodes)
        s=Search(game); s.alphabeta(start,d); ab.append(s.nodes)
        s=Search(game); _,timings=s.ids(start,d); ids.append(sum(t[1] for t in timings))

    plt.figure(figsize=(8,5)); plt.plot(list(depths),mm,marker="o",label="MiniMax"); plt.plot(list(depths),ab,marker="o",label="Alpha-Beta"); plt.plot(list(depths),ids,marker="o",label="Alpha-Beta + IDS"); plt.xlabel("Depth"); plt.ylabel("Nodes explored"); plt.title("Dock contention search comparison"); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()

    return list(zip(depths,mm,ab,ids))

def play(depth=5,opponent="random"):
    game=DockGame(); state=game.piles; ai_turn=True

    while not game.terminal(state):
        print("Piles:",state)
        if ai_turn:
            search=Search(game); move,timings=search.ids(state,depth); chosen=move; print("AI move:", chosen[:2], "nodes:", timings[-1][1])
        elif opponent=="random":
            chosen=random.choice(game.moves(state)); print("Opponent move:",chosen[:2])
        else:
            search=Search(game); _,chosen=search.alphabeta(state,depth,maximizing=True); print("Opponent AI move:",chosen[:2])

        state=chosen[2]; ai_turn=not ai_turn

    print("Dock claimed by:", "AI" if not ai_turn else "Opponent")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--depth",type=int,default=5); parser.add_argument("--self-play",action="store_true"); parser.add_argument("--chart",action="store_true"); args=parser.parse_args()

    if args.chart:
        rows=chart(); print("depth minimax alpha_beta alpha_beta_ids"); [print(*r) for r in rows]
    play(args.depth,"self" if args.self_play else "random")

if __name__ == "__main__": main()