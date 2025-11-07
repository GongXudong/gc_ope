
import numpy as np
from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer, WeightedEvaluationResultContainer


def test_evaluation_result_container_init():
    container = EvaluationResultContainer()

    assert container.desired_goal_list == []
    assert container.success_list == []
    assert container.cumulative_reward_list == []
    assert container.discounted_cumulative_reward_list == []

def test_evaluation_result_container_add():

    desired_goal_list = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
    ]
    success_list = [True, False, False, True, True]
    cumulative_reward_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    discounted_cumulative_reward_list = [10.0, 20.0, 30.0, 40.0, 50.0]

    container = EvaluationResultContainer()
    
    for index, (dg, success, cr, dcr) in enumerate(zip(desired_goal_list, success_list, cumulative_reward_list, discounted_cumulative_reward_list)):
        container.add(dg, success, cr, dcr)

        assert np.allclose(container.desired_goal_list, desired_goal_list[:index+1])
        assert np.allclose(container.success_list, success_list[:index+1])
        assert np.allclose(container.cumulative_reward_list, cumulative_reward_list[:index+1])
        assert np.allclose(container.discounted_cumulative_reward_list, discounted_cumulative_reward_list[:index+1])

def test_evaluation_result_container_add_batch():
    desired_goal_batch_list = [[
        [1, 1, 1],
        [2, 2, 2],
    ], [
        [3, 3, 3],
        [4, 4, 4],
    ],[
        [5, 5, 5],
        [6, 6, 6],
    ]]
    success_batch_list = [[True, False], [False, True], [True, False]]
    cumulative_reward_batch_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    discounted_cumulative_reward_batch_list = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]

    container = EvaluationResultContainer()
    
    for index, (dg_batch, success_batch, cr_batch, dcr_batch) in enumerate(zip(desired_goal_batch_list, success_batch_list, cumulative_reward_batch_list, discounted_cumulative_reward_batch_list)):
        container.add_batch(dg_batch, success_batch, cr_batch, dcr_batch)

        assert np.allclose(container.desired_goal_list, np.array(desired_goal_batch_list[:index+1]).reshape([-1, 3]))
        assert np.allclose(container.success_list, np.array(success_batch_list[:index+1]).reshape([-1]))
        assert np.allclose(container.cumulative_reward_list, np.array(cumulative_reward_batch_list[:index+1]).reshape([-1]))
        assert np.allclose(container.discounted_cumulative_reward_list, np.array(discounted_cumulative_reward_batch_list[:index+1]).reshape([-1]))

def test_weighted_evaluation_result_container_init():
    discount_factor = 0.9
    container = WeightedEvaluationResultContainer(discounted_factor=discount_factor)

    assert container.desired_goal_list == []
    assert container.success_list == []
    assert container.cumulative_reward_list == []
    assert container.discounted_cumulative_reward_list == []
    assert container.desired_goal_weights.size == 0

def test_weighted_evaluation_result_container_add():

    desired_goal_list = [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [5, 5, 5],
    ]
    success_list = [True, False, False, True, True]
    cumulative_reward_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    discounted_cumulative_reward_list = [10.0, 20.0, 30.0, 40.0, 50.0]

    discount_factor = 0.9
    container = WeightedEvaluationResultContainer(discounted_factor=discount_factor)

    print(container.desired_goal_weights, container.desired_goal_list, container.success_list, container.cumulative_reward_list, container.discounted_cumulative_reward_list)
    
    for index, (dg, success, cr, dcr) in enumerate(zip(desired_goal_list, success_list, cumulative_reward_list, discounted_cumulative_reward_list)):
        container.add(dg, success, cr, dcr, 1.0)

        assert np.allclose(container.desired_goal_list, desired_goal_list[:index+1])
        assert np.allclose(container.success_list, success_list[:index+1])
        assert np.allclose(container.cumulative_reward_list, cumulative_reward_list[:index+1])
        assert np.allclose(container.discounted_cumulative_reward_list, discounted_cumulative_reward_list[:index+1])
        
        weights = np.power(discount_factor, np.arange(index + 1)[::-1])
        
        print(container.desired_goal_weights)
        print(weights)

        assert np.allclose(container.desired_goal_weights, weights)

def test_weighted_evaluation_result_container_add_batch():
    desired_goal_batch_list = [[
        [1, 1, 1],
        [2, 2, 2],
    ], [
        [3, 3, 3],
        [4, 4, 4],
    ],[
        [5, 5, 5],
        [6, 6, 6],
    ]]
    success_batch_list = [[True, False], [False, True], [True, False]]
    cumulative_reward_batch_list = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    discounted_cumulative_reward_batch_list = [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    weights = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]

    discount_factor = 0.9
    container = WeightedEvaluationResultContainer(discounted_factor=discount_factor)
    
    for index, (dg_batch, success_batch, cr_batch, dcr_batch, weight_batch) in enumerate(zip(desired_goal_batch_list, success_batch_list, cumulative_reward_batch_list, discounted_cumulative_reward_batch_list, weights)):
        container.add_batch(dg_batch, success_batch, cr_batch, dcr_batch, weight_batch)

        assert np.allclose(container.desired_goal_list, np.array(desired_goal_batch_list[:index+1]).reshape([-1, 3]))
        assert np.allclose(container.success_list, np.array(success_batch_list[:index+1]).reshape([-1]))
        assert np.allclose(container.cumulative_reward_list, np.array(cumulative_reward_batch_list[:index+1]).reshape([-1]))
        assert np.allclose(container.discounted_cumulative_reward_list, np.array(discounted_cumulative_reward_batch_list[:index+1]).reshape([-1]))

        weights_calc = np.power(discount_factor, np.arange(index + 1)[::-1])

        weights_new = np.array(weights[:index+1]).copy()
        for ii in range(index+1):
            weights_new[ii] *= weights_calc[ii]

        print(weights_new)
        
        assert np.allclose(container.desired_goal_weights, weights_new.reshape([-1]))

if __name__ == "__main__":
    # test_evaluation_result_container_add()
    # test_evaluation_result_container_add_batch()
    # test_weighted_evaluation_result_container_add()
    test_weighted_evaluation_result_container_add_batch()
