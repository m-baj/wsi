import pytest
from minimax import State, calculate_value_for, minimax


def test_state():
    state = State(10, True, 3)
    assert state.tokens_left == 10
    assert state.is_max_player_move
    assert not state.is_terminal

    state = State(0, False, 3)
    assert state.tokens_left == 0
    assert not state.is_max_player_move
    assert state.is_terminal


def test_get_child_states():
    state = State(10, True, 3)
    children_states = state.get_child_states()
    assert len(children_states) == 3
    assert children_states[0].tokens_left == 9
    assert children_states[0].is_max_player_move == False
    assert children_states[0].is_terminal == False

    assert children_states[1].tokens_left == 8
    assert children_states[1].is_max_player_move == False
    assert children_states[1].is_terminal == False

    assert children_states[2].tokens_left == 7
    assert children_states[2].is_max_player_move == False
    assert children_states[2].is_terminal == False
