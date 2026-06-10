# RRT-Connect 规划算子

在关节空间采样，以 `CollisionModel` 做构型/边可行性检验。

## 伪代码

```
tree_a <- {q_start}, tree_b <- {q_goal}
repeat max_iterations:
    q_rand <- sample with goal_bias
    q_new <- steer(nearest(tree_a, q_rand))
    if edge_collision_free(q_near, q_new):
        add q_new to tree_a
        try connect to tree_b
        if connected: return path
    swap tree_a, tree_b
```

## API

`RRTConnectPlanner(runtime, collision_model, step_size, goal_bias, max_iterations, edge_collision_samples)`
