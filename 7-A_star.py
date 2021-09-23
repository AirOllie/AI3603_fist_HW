import DR20API
import numpy as np
import heapq

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.

from queue import PriorityQueue as PQ
from matplotlib import pyplot as plt
fringe = []
closed_set = np.zeros((120, 120), bool)
cost = np.zeros((120, 120))
start_pos = np.zeros((2,))
# indicating the last grid of [i, j] grid
father = np.zeros((120, 120, 2), int)
direction = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
goal_pos = np.array([100, 100])
GAMMA = 1.01


class Point:
    def __init__(self, pos) -> None:
        self.x, self.y = pos
        self.pos = pos

    def __lt__(self, other):
        if total_cost(self.pos, goal_pos) < total_cost(other.pos, goal_pos):
            return True
        else:
            return False

    def __eq__(self, o: object) -> bool:
        return (o.pos == self.pos).all() and self.x == o.x and self.y == o.y

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)


def total_cost(current_pos: np.ndarray, goal_pos: np.ndarray):
    current_pos = np.asarray(current_pos)
    goal_pos = np.asarray(goal_pos)
    x, y = current_pos
    return cost[x, y] + heuristic(current_pos, goal_pos) * GAMMA


def heuristic(current_pos: np.ndarray, goal_pos: np.ndarray):
    """return the Manhattan distance from current pos to goal pos

    Args:
        current_pos (np.ndarray): A 2D vector indicating the current position of the robot.
        goal_pos (np.ndarray): A 2D vector indicating the position of the goal.

    Returns:
        [int]: [the Manhattan distance from current pos to goal pos]
    """
    res = np.linalg.norm(current_pos - goal_pos, 1)
    return res


def draw_path(path, current_map):
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if current_map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(current_pos[0], current_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def is_valid_point(current_pos: np.ndarray, current_map: np.ndarray):
    x, y = current_pos
    if x < 0 or x >= 120 or y < 0 or y >= 120:
        return False
    return not current_map[x, y]


def draw_father():
    plt.figure()
    ax = plt.gca()
    indices = np.argwhere(np.any((father != 0), -1))
    for i, j in indices:
        ax.arrow(i, j, father[i, j, 0] - i, father[i, j, 1] - j, width=0.01,
                 length_includes_head=True,  # 增加的长度包含箭头部分
                 head_width=0.2,
                 head_length=0.5,
                 fc='r',
                 ec='b')

    plt.show()


def build_path(current_pos: np.ndarray, original_pos: np.ndarray):
    path = []
    start = current_pos[:]
    end = original_pos[:]

    draw_father()
    path.insert(0, start)
    while heuristic(start, end) > 2:
        start = father[start[0], start[1]]
        path.insert(0, start)
    path.insert(0, end)
    return path


###  END CODE HERE  ###


def A_star(current_map, current_pos, goal_pos):
    """
    Given current map of the world, current position of the robot and the position of the goal, 
    plan a path from current position to the goal using A* algorithm.

    Arguments:
    current_map -- A 120*120 array indicating current map, where 0 indicating traversable and 1 indicating obstacles.
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by A* algorithm.
    """

    ### START CODE HERE ###
    current_map = np.asarray(current_map)
    current_pos = np.asarray(current_pos)
    goal_pos = np.asarray(goal_pos) + 1
    if Point(current_pos) not in fringe:

        heapq.heappush(fringe, Point(current_pos))
    # path = []
    cnt = 0
    while fringe:

        # print(np.argwhere(father))

        temp = heapq.heappop(fringe)
        cnt += 1
        x, y = temp.x, temp.y
        # already visited and pop from the fringe, add it to the close set
        closed_set[x, y] = True
        if reach_goal(temp.pos, goal_pos) or heuristic(temp.pos, current_pos) >= 29:
            path = build_path(temp.pos, current_pos)
            break

        pax, pay = father[x, y]
        # print(x, y, pax, pay)
        if not np.array_equal(temp.pos, start_pos):
            cost[x, y] = cost[pax, pay] + 1
        for i in range(4):
            new_pos = temp.pos + direction[i]
            newx, newy = new_pos
            if is_valid_point(new_pos, current_map):
                if closed_set[newx, newy]:
                    # if cost[x, y] + 1 < cost[new_pos, goal_pos]:
                    #     cost[new_pos, goal_pos] = cost[x, y] + 1
                    #     father[newx, newy] = temp
                    continue
                if Point(new_pos) in fringe:
                    if cost[x, y] + 1 <= cost[newx, newy]:
                        # find a shorter path wiht fewer f(x)
                        cost[newx, newy] = cost[x, y] + 1
                        father[newx, newy] = temp.pos
                        heapq.heapify(fringe)
                else:
                    cost[newx, newy] = cost[x, y] + 1
                    father[newx, newy] = temp.pos
                    heapq.heappush(fringe, Point(new_pos))

        # if the final goal is popped from the fringe, the search is over

    ###  END CODE HERE  ###
    return path


def reach_goal(current_pos, goal_pos):
    """
    Given current position of the robot, 
    check whether the robot has reached the goal.

    Arguments:
    current_pos -- A 2D vector indicating the current position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    is_reached -- A bool variable indicating whether the robot has reached the goal, where True indicating reached.
    """

    ### START CODE HERE ###
    is_reached = heuristic(current_pos, goal_pos) <= 2

    ###  END CODE HERE  ###
    return is_reached


if __name__ == '__main__':
    # Define goal position of the exploration, shown as the gray block in the scene.

    goal_pos = [100, 100]
    controller = DR20API.Controller()

    # Initialize the position of the robot and the map of the world.
    current_pos = controller.get_robot_pos()
    current_map = controller.update_map()
    start_pos = current_pos[:]

    # print(fringe.get())
    # Plan-Move-Perceive-Update-Replan loop until the robot reaches the goal.
    while not reach_goal(current_pos, goal_pos):
        # Plan a path based on current map from current position of the robot to the goal.
        path = A_star(current_map, current_pos, goal_pos)
        print(path[:30])
        # draw_path(path, current_map)
        # Move the robot along the path to a certain distance.
        controller.move_robot(path)
        # Get current position of the robot.
        current_pos = controller.get_robot_pos()
        # Update the map based on the current information of laser scanner and get the updated map.
        current_map = controller.update_map()

    # Stop the simulation.
    controller.stop_simulation()
