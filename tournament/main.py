import sys
import os

# Ensure the project root is in sys.path so we can import from tournament
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tournament.core import Player, Match

def calculate_probabilities():
    # Define players
    bo = Player(name="Bo")
    ci = Player(name="Ci")
    al = Player(name="Al")

    # Probabilities
    p_bo_beats_ci = 0.6
    p_ci_beats_bo = 1.0 - p_bo_beats_ci # 0.4

    p_al_beats_bo = 0.5
    p_bo_beats_al = 1.0 - p_al_beats_bo # 0.5

    p_al_beats_ci = 0.7
    p_ci_beats_al = 1.0 - p_al_beats_ci # 0.3

    # Define Matches
    # Round 1: Bo vs Ci. 2 games. condition: win both.
    match_r1 = Match(challenger=bo, defender=ci, num_games=2)

    # Round 2 (if Bo wins R1): Bo vs Al. 2 games. condition: win both. Short circuit if Al wins G1.
    match_r2_bo = Match(challenger=bo, defender=al, num_games=2)

    # Round 2 (if Ci wins R1): Ci vs Al. 2 games. condition: win both. Short circuit if Al wins G1.
    match_r2_ci = Match(challenger=ci, defender=al, num_games=2)

    # 1. A priori probabilities

    # (b) Bo will win the first round.
    # Bo wins R1 if Bo wins both games against Ci.
    prob_bo_wins_r1 = match_r1.prob_challenger_wins_match(p_bo_beats_ci)

    # Ci will win the first round.
    # Ci is the defender in match_r1 context but logic is symmetric.
    # Ci wins R1 if Ci wins both games against Bo.
    prob_ci_wins_r1 = p_ci_beats_bo ** 2

    # (a) The second round will be required.
    # Required if Bo wins R1 OR Ci wins R1. Mutually exclusive?
    # Bo wins 2 games -> Bo wins R1.
    # Ci wins 2 games -> Ci wins R1.
    # 1-1 split -> Neither wins R1.
    # So yes, mutually exclusive because one cannot both win 2 games in a 2-game match.
    prob_2nd_round_req = prob_bo_wins_r1 + prob_ci_wins_r1

    # (c) Al will retain his championship this year.
    # Al retains if R2 is NOT required OR (R2 required AND Challenger loses R2).
    # P(Al retains) = 1 - P(Challenger wins championship)
    # Challenger wins championship if (Bo wins R1 AND Bo wins R2) OR (Ci wins R1 AND Ci wins R2)
    prob_bo_champ = prob_bo_wins_r1 * match_r2_bo.prob_challenger_wins_match(p_bo_beats_al)
    prob_ci_champ = prob_ci_wins_r1 * match_r2_ci.prob_challenger_wins_match(p_ci_beats_al)
    prob_al_champ = 1.0 - (prob_bo_champ + prob_ci_champ)

    print(f"1(a) P(2nd Round Req) = {prob_2nd_round_req:.4f}")
    print(f"1(b) P(Bo Wins 1st Round) = {prob_bo_wins_r1:.4f}")
    print(f"1(c) P(Al Champ) = {prob_al_champ:.4f}")

    # 2. Conditional probabilities given 2nd round is required

    # (a) Bo is the surviving challenger.
    # P(Bo Wins R1 | 2nd Round Req) = P(Bo Wins R1 AND 2nd Round Req) / P(2nd Round Req)
    # Since Bo Wins R1 implies 2nd Round Req, numerator is P(Bo Wins R1)
    prob_bo_challenger_given_r2 = prob_bo_wins_r1 / prob_2nd_round_req

    # For calculation of (b), we also need P(Ci Challenger | 2nd Round Req)
    prob_ci_challenger_given_r2 = prob_ci_wins_r1 / prob_2nd_round_req

    print(f"2(a) P(Bo Challenger | 2nd Round Req) = {prob_bo_challenger_given_r2:.4f}")

    # (b) Al retains his championship.
    # P(Al Champ | 2nd Round Req)
    # = P(Al Champ AND 2nd Round Req) / P(2nd Round Req)
    # Al Champ AND 2nd Round Req means:
    # (Bo Wins R1 AND Bo loses R2) OR (Ci Wins R1 AND Ci loses R2)
    prob_bo_wins_r1_and_loses_r2 = prob_bo_wins_r1 * (1.0 - match_r2_bo.prob_challenger_wins_match(p_bo_beats_al))
    prob_ci_wins_r1_and_loses_r2 = prob_ci_wins_r1 * (1.0 - match_r2_ci.prob_challenger_wins_match(p_ci_beats_al))

    prob_al_champ_given_r2 = (prob_bo_wins_r1_and_loses_r2 + prob_ci_wins_r1_and_loses_r2) / prob_2nd_round_req

    print(f"2(b) P(Al Champ | 2nd Round Req) = {prob_al_champ_given_r2:.4f}")

    # 3. Given that the second round was required and that it comprised only one game, what is the conditional probability that it was Bo who won the first round?
    # P(Bo Challenger | 2nd Round Req AND R2 Length == 1)
    # = P(Bo Challenger AND R2 Length == 1) / P(2nd Round Req AND R2 Length == 1)
    # Note: "2nd Round Req" is implied by "R2 Length == 1" (can't have length 1 if not required).
    # Actually, be careful. "R2 Length == 1" implies R2 happened.

    # Numerator: Bo Challenger (Bo Wins R1) AND R2 Length == 1 (in Bo vs Al match)
    # R2 Length == 1 means Al wins the first game.
    # P(Bo Wins R1) * P(R2 Length == 1 | Bo Challenger)
    # match_r2_bo length 1: Al wins G1.
    prob_r2_bo_len_1 = match_r2_bo.prob_match_length(1, p_bo_beats_al, short_circuit_on_defender_win=True)
    prob_numerator = prob_bo_wins_r1 * prob_r2_bo_len_1

    # Denominator: P(R2 Length == 1)
    # = P(Bo Challenger AND R2 Bo Length 1) + P(Ci Challenger AND R2 Ci Length 1)
    prob_r2_ci_len_1 = match_r2_ci.prob_match_length(1, p_ci_beats_al, short_circuit_on_defender_win=True)
    prob_denominator = (prob_bo_wins_r1 * prob_r2_bo_len_1) + (prob_ci_wins_r1 * prob_r2_ci_len_1)

    prob_q3 = prob_numerator / prob_denominator

    print(f"3. P(Bo Challenger | 2nd Round Req AND One Game) = {prob_q3:.4f}")

if __name__ == "__main__":
    calculate_probabilities()
