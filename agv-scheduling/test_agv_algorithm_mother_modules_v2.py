import unittest

from agv_algorithm_mother_modules_v2 import (
    AGV,
    Task,
    WeightedGraph,
    ProblemDefinition,
    SchedulingSnapshot,
    CandidateGenerator,
    ConstraintChecker,
    ObjectiveEvaluator,
    RuleDecisionEngine,
    ReservationTable,
    ReplayBuffer,
    Transition,
    TabularQLearner,
    HybridCoordinator,
    SafetyShield,
    ActionMask,
    RiskEvaluator,
    MultiObjectiveEvaluator,
    ModelBasedPlanner,
    GameCoordinator,
    FeaturePipeline,
    HierarchicalController,
)


def build_problem():
    graph = WeightedGraph()
    graph.add_edge("A", "B", 1.0)
    graph.add_edge("B", "C", 1.0)
    graph.add_edge("A", "D", 2.0)
    graph.add_edge("D", "C", 2.0)
    return ProblemDefinition(graph=graph, min_battery=15.0, energy_per_distance=1.0)


class TestReusableMotherModules(unittest.TestCase):
    def test_candidate_generation_and_constraints(self):
        problem = build_problem()
        snapshot = SchedulingSnapshot(
            time=0,
            agvs=(
                AGV("agv-1", "A", battery=90.0, capacity=2.0),
                AGV("agv-2", "C", battery=10.0, capacity=2.0),
                AGV("agv-3", "A", battery=90.0, capacity=2.0, busy=True),
            ),
            tasks=(Task("task-1", "B", "C", load=1.0),),
        )
        raw = CandidateGenerator().generate(snapshot)
        feasible = ConstraintChecker().filter(problem, snapshot, raw)
        self.assertEqual([(x.agv_id, x.task_id) for x in feasible], [("agv-1", "task-1")])

    def test_rule_engine_selects_lowest_cost_candidate(self):
        problem = build_problem()
        snapshot = SchedulingSnapshot(
            time=0,
            agvs=(AGV("near", "A", 90.0, 2.0), AGV("far", "D", 90.0, 2.0)),
            tasks=(Task("task-1", "B", "C", 1.0, priority=2.0),),
        )
        candidates = ConstraintChecker().filter(
            problem, snapshot, CandidateGenerator().generate(snapshot)
        )
        decision = RuleDecisionEngine(ObjectiveEvaluator()).decide(problem, snapshot, candidates)
        self.assertEqual(decision.agv_id, "near")

    def test_shortest_path(self):
        problem = build_problem()
        path, cost = problem.graph.shortest_path("A", "C")
        self.assertEqual(path, ["A", "B", "C"])
        self.assertEqual(cost, 2.0)

    def test_reservation_table_repairs_node_conflict_by_waiting(self):
        reservations = ReservationTable()
        reservations.reserve(["X", "B"], start_time=0, owner="other")
        repaired, waits = reservations.resolve(["A", "B", "C"], start_time=0)
        self.assertEqual(repaired, ["A", "A", "B", "C"])
        self.assertEqual(waits, 1)

    def test_replay_buffer_capacity_and_sampling(self):
        buffer = ReplayBuffer(capacity=2, seed=7)
        for i in range(3):
            buffer.add(Transition(i, 0, float(i), i + 1, False))
        self.assertEqual(len(buffer), 2)
        sample = buffer.sample(2)
        self.assertEqual({x.state for x in sample}, {1, 2})

    def test_q_learning_update(self):
        learner = TabularQLearner(alpha=0.5, gamma=0.9, seed=1)
        learner.set_q("next", "go", 10.0)
        td_error = learner.update("now", "go", reward=1.0, next_state="next", next_actions=["go"])
        self.assertAlmostEqual(td_error, 10.0)
        self.assertAlmostEqual(learner.get_q("now", "go"), 5.0)

    def test_hybrid_coordinator_produces_valid_plan(self):
        problem = build_problem()
        snapshot = SchedulingSnapshot(
            time=0,
            agvs=(AGV("agv-1", "A", battery=90.0, capacity=2.0),),
            tasks=(Task("task-1", "B", "C", load=1.0),),
        )
        coordinator = HybridCoordinator(
            decision_engine=RuleDecisionEngine(ObjectiveEvaluator()),
            safety_shield=SafetyShield(),
        )
        plan = coordinator.plan_one(problem, snapshot)
        self.assertEqual(plan.agv_id, "agv-1")
        self.assertEqual(plan.task_id, "task-1")
        self.assertEqual(plan.full_path, ["A", "B", "C"])
        self.assertTrue(plan.safe)

    def test_action_mask_keeps_only_feasible_candidates(self):
        all_candidates = [
            ("agv-1", "task-1"),
            ("agv-2", "task-1"),
            ("agv-3", "task-1"),
        ]
        mask = ActionMask.from_keys(all_candidates, {("agv-1", "task-1"), ("agv-3", "task-1")})
        self.assertEqual(mask.allowed, (True, False, True))
        self.assertEqual(mask.legal_indices(), [0, 2])
        with self.assertRaises(ValueError):
            mask.validate(1)

    def test_risk_and_multiobjective_evaluators(self):
        risk = RiskEvaluator()
        self.assertEqual(risk.aggregate([1.0, 2.0, 9.0], mode="worst"), 9.0)
        self.assertEqual(risk.aggregate([1.0, 2.0, 9.0], mode="mean"), 4.0)
        evaluator = MultiObjectiveEvaluator(minimize=("distance", "energy"))
        records = [
            {"name": "a", "distance": 5.0, "energy": 5.0},
            {"name": "b", "distance": 4.0, "energy": 7.0},
            {"name": "c", "distance": 6.0, "energy": 6.0},
        ]
        self.assertEqual({x["name"] for x in evaluator.pareto_front(records)}, {"a", "b"})

    def test_model_based_planner_looks_ahead(self):
        planner = ModelBasedPlanner(
            action_provider=lambda _state: [1, 2],
            transition_model=lambda state, action: state + action,
            reward_model=lambda _state, _action, next_state: float(next_state),
            horizon=2,
            discount=1.0,
            beam_width=4,
        )
        action, score, sequence = planner.plan(0)
        self.assertEqual(action, 2)
        self.assertEqual(sequence, [2, 2])
        self.assertEqual(score, 6.0)

    def test_game_coordinator_and_feature_pipeline(self):
        game = GameCoordinator(
            players=("a", "b"),
            action_provider=lambda _player, _state: [0, 1],
            utility=lambda player, action, joint, _state: 1.0
            if action == joint["b" if player == "a" else "a"]
            else 0.0,
        )
        result = game.best_response(state=None, initial={"a": 0, "b": 1}, rounds=3)
        self.assertEqual(result, {"a": 1, "b": 1})
        pipeline = FeaturePipeline([lambda x: x + 1, lambda x: x * 10])
        self.assertEqual(pipeline.transform(2), 30)
        hierarchy = HierarchicalController(
            high_level_policy=lambda state: f"serve-{state}",
            low_level_policy=lambda state, goal: (state, goal, "move"),
        )
        self.assertEqual(
            hierarchy.decide("task-1"),
            ("serve-task-1", ("task-1", "serve-task-1", "move")),
        )


if __name__ == "__main__":
    unittest.main()
