import pytest
from tournament.core import Player, Match

def test_match_win_probability():
    p1 = Player("P1")
    p2 = Player("P2")
    match = Match(p1, p2, num_games=2)

    # p = 0.5. Win all 2 games = 0.25
    assert match.prob_challenger_wins_match(0.5) == 0.25
    assert match.prob_defender_wins_match(0.5) == 0.75

    # p = 1.0. Win all 2 = 1.0
    assert match.prob_challenger_wins_match(1.0) == 1.0
    assert match.prob_defender_wins_match(1.0) == 0.0

    # p = 0.0. Win all 2 = 0.0
    assert match.prob_challenger_wins_match(0.0) == 0.0
    assert match.prob_defender_wins_match(0.0) == 1.0

def test_match_length_no_short_circuit():
    p1 = Player("P1")
    p2 = Player("P2")
    match = Match(p1, p2, num_games=3)

    # Without short circuit, length is always num_games
    assert match.prob_match_length(3, 0.5, short_circuit_on_defender_win=False) == 1.0
    assert match.prob_match_length(2, 0.5, short_circuit_on_defender_win=False) == 0.0

def test_match_length_with_short_circuit():
    p1 = Player("P1")
    p2 = Player("P2")
    match = Match(p1, p2, num_games=2)

    # Short circuit on defender win (defender wins game -> match ends)
    # p_challenger = 0.6
    # p_defender = 0.4

    # Length 1: Defender wins G1. Prob 0.4.
    assert match.prob_match_length(1, 0.6, short_circuit_on_defender_win=True) == pytest.approx(0.4)

    # Length 2: Challenger wins G1 (0.6). Match continues.
    # It ends at 2 regardless of G2 outcome (because num_games=2).
    # So Prob = 0.6.
    assert match.prob_match_length(2, 0.6, short_circuit_on_defender_win=True) == pytest.approx(0.6)

    # Sum should be 1.0
    assert match.prob_match_length(1, 0.6, True) + match.prob_match_length(2, 0.6, True) == pytest.approx(1.0)

def test_match_length_short_circuit_3_games():
    p1 = Player("P1")
    p2 = Player("P2")
    match = Match(p1, p2, num_games=3)
    p = 0.5

    # Length 1: Def wins G1. (0.5)
    assert match.prob_match_length(1, p, True) == 0.5

    # Length 2: Chall wins G1 (0.5) * Def wins G2 (0.5) = 0.25
    assert match.prob_match_length(2, p, True) == 0.25

    # Length 3: Chall wins G1 (0.5) * Chall wins G2 (0.5) = 0.25. (Ends at 3)
    assert match.prob_match_length(3, p, True) == 0.25

    assert 0.5 + 0.25 + 0.25 == 1.0

def test_chess_problem_regression():
    # Verify the specific answers from the problem
    bo = Player("Bo")
    ci = Player("Ci")
    al = Player("Al")

    p_bo_beats_ci = 0.6
    p_ci_beats_bo = 0.4 # implied

    match_r1 = Match(bo, ci, 2)

    # 1(a)
    prob_bo_wins_r1 = match_r1.prob_challenger_wins_match(p_bo_beats_ci)
    prob_ci_wins_r1 = p_ci_beats_bo ** 2
    prob_2nd_round_req = prob_bo_wins_r1 + prob_ci_wins_r1
    assert prob_2nd_round_req == pytest.approx(0.52)

    # 1(b)
    assert prob_bo_wins_r1 == pytest.approx(0.36)

    # 1(c)
    p_bo_beats_al = 0.5
    p_ci_beats_al = 0.3
    match_r2_bo = Match(bo, al, 2)
    match_r2_ci = Match(ci, al, 2)

    prob_bo_champ = prob_bo_wins_r1 * match_r2_bo.prob_challenger_wins_match(p_bo_beats_al)
    prob_ci_champ = prob_ci_wins_r1 * match_r2_ci.prob_challenger_wins_match(p_ci_beats_al)
    prob_al_champ = 1.0 - (prob_bo_champ + prob_ci_champ)

    assert prob_al_champ == pytest.approx(0.8956)

    # 2(a)
    assert prob_bo_wins_r1 / prob_2nd_round_req == pytest.approx(0.6923, abs=1e-4)

    # 2(b)
    prob_bo_wins_r1_loses_r2 = prob_bo_wins_r1 * (1.0 - match_r2_bo.prob_challenger_wins_match(p_bo_beats_al))
    prob_ci_wins_r1_loses_r2 = prob_ci_wins_r1 * (1.0 - match_r2_ci.prob_challenger_wins_match(p_ci_beats_al))
    prob_al_champ_given_r2 = (prob_bo_wins_r1_loses_r2 + prob_ci_wins_r1_loses_r2) / prob_2nd_round_req

    assert prob_al_champ_given_r2 == pytest.approx(0.7992, abs=1e-4)

    # 3
    prob_r2_bo_len_1 = match_r2_bo.prob_match_length(1, p_bo_beats_al, short_circuit_on_defender_win=True)
    prob_r2_ci_len_1 = match_r2_ci.prob_match_length(1, p_ci_beats_al, short_circuit_on_defender_win=True)

    prob_num = prob_bo_wins_r1 * prob_r2_bo_len_1
    prob_den = (prob_bo_wins_r1 * prob_r2_bo_len_1) + (prob_ci_wins_r1 * prob_r2_ci_len_1)

    assert prob_num / prob_den == pytest.approx(0.6164, abs=1e-4)
