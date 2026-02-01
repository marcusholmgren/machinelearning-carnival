from dataclasses import dataclass

@dataclass
class Player:
    name: str

class Match:
    def __init__(self, challenger: Player, defender: Player, num_games: int):
        self.challenger = challenger
        self.defender = defender
        self.num_games = num_games

    def prob_challenger_wins_match(self, p_challenger_wins_game: float) -> float:
        """
        Calculates probability that challenger wins the match.
        For this tournament type, the challenger must win all games to win the match.
        """
        return p_challenger_wins_game ** self.num_games

    def prob_defender_wins_match(self, p_challenger_wins_game: float) -> float:
        """
        Calculates probability that defender wins the match.
        The defender retains their position (wins the match) if the challenger fails to win all games.
        """
        return 1.0 - self.prob_challenger_wins_match(p_challenger_wins_game)

    def prob_match_length(self, length: int, p_challenger_wins_game: float, short_circuit_on_defender_win: bool = False) -> float:
        """
        Calculates the probability that the match lasts exactly `length` games.

        Args:
            length: The number of games.
            p_challenger_wins_game: Probability challenger wins a single game.
            short_circuit_on_defender_win: If True, match ends immediately if defender wins a game.
                                           (Because challenger needs to win ALL games, so one loss makes it impossible).
        """
        if not short_circuit_on_defender_win:
            # If no short circuit, match always lasts num_games (unless we define other rules, but for now fixed length)
            # Assuming fixed length unless short circuit.
            return 1.0 if length == self.num_games else 0.0

        # With short circuit on defender win:
        # Match ends at game k < num_games if:
        #   Challenger won games 1 to k-1 (prob p^(k-1))
        #   AND Defender wins game k (prob 1-p)
        # Match ends at game num_games if:
        #   Challenger won games 1 to num_games-1 (prob p^(n-1))
        #   (The result of the last game doesn't matter for length, it ends anyway)

        if length < 1 or length > self.num_games:
            return 0.0

        if length < self.num_games:
            # Ends early implies defender won this specific game, and challenger won previous ones.
            return (p_challenger_wins_game ** (length - 1)) * (1.0 - p_challenger_wins_game)

        if length == self.num_games:
            # Reached the end. Means challenger won all previous n-1 games.
            return p_challenger_wins_game ** (self.num_games - 1)

        return 0.0
