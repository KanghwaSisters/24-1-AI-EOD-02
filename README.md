# 24-1-AI-EOD-02
[ 24-1 /  AI EOD / Team 02 ]  
👩‍💻 이은나, 김정은

# 환경 (김정은)

- 사용 라이브러리

```python
import numpy as np
import pandas as pd
from typing import Tuple
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import gym
from gym import spaces
import numpy as np
import os
import time
```

### Init

```python
class MinesweeperEnv(gym.Env):
    def __init__(self, board_size=9, num_mines=10):
        super(MinesweeperEnv, self).__init__()

        self.board_size = board_size
        self.num_mines = num_mines

        # 행동 공간 및 관측 공간 정의
        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(low=0, high=2, shape=(2, board_size, board_size), dtype=np.int64)  # Channels, Height, Width

        self.reset()
```

1. **`__init__` 메서드**: 환경 초기 설정을 담당. 보드 크기, 지뢰 개수, 행동 공간, 관측 공간을 정의.

( board_size는 9x9의 공간, 지뢰 개수는 10개)

`action_space`는 `gym`  을 활용하여 `spaces.Discrete` 로 구현, 행동 공간 정의 (보드 크기 * 보드 크기 개수만큼의 행동)

`observation_space`는 `gym`을 활용하여 `spaces.Box`로 구현, 관측 공간 정의 (보드 크기, 보드 크기, 2) 크기의 박스, 값은 0에서 2까지(`shape`을 보면 이는 3차원 배열)

- 관측 공간은 **`spaces.Box(low=0, high=2, shape=(board_size, board_size, 2), dtype=np.int64)`**로 정의되어 있음. 이는 보드의 각 셀이 두 개의 값(열렸는지 여부와 인접한 지뢰의 수)으로 구성된 상태를 가집니다.
- **`low=0`**과 **`high=2`**는 관측 공간의 각 값이 가질 수 있는 최소값과 최대값을 정의. 여기서 **`high=2`**는 각 값이 0, 1, 2 중 하나가 될 수 있음을 의미. 이는 상태가 두 개의 값으로 구성되어 있기 때문에 필요함
    - 첫 번째 값은 셀이 열렸는지(0 또는 1).
    - 두 번째 값은 인접한 지뢰의 수(0에서 8까지, 하지만 최대값을 2로 제한하여 간단하게 표현)

### Reset

```python
def reset(self):
        # 보드와 상태 초기화
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.state = np.zeros((2, self.board_size, self.board_size), dtype=int)  # Channels, Height, Width

        # 지뢰 위치 설정
        self.mines = np.random.choice(self.board_size * self.board_size, self.num_mines, replace=False)
        for mine in self.mines:
            x, y = divmod(mine, self.board_size)
            self.board[x, y] = -1

        # 인접한 지뢰 개수 계산
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == -1:
                    continue
                count = 0
                for x in range(max(0, i - 1), min(self.board_size, i + 2)):
                    for y in range(max(0, j - 1), min(self.board_size, j + 2)):
                        if self.board[x, y] == -1:
                            count += 1
                self.board[i, j] = count

        # 초기 상태 설정
        self.done = False
        self.steps = 0
        self.total_reward = 0  # 누적 보상 초기화
        self.mine_hit = False
        self.first_click = True

        return self._get_observation()
```

2.  **`reset` 메서드**: 환경을 초기화하고, 게임 보드와 상태를 초기 상태로 설정. 지뢰를 무작위로 배치하고 각 셀에 인접한 지뢰 개수를 계산합니다.

- 먼저, `Board`와 `State`를 모두 초기화, `Numpy`를 사용하여 0으로 설정.
- `np.random.choice` 를 이용하여 무작위로 지뢰 위치 선택
- `self.board[x, y] = -1`  지뢰 위치에 -1 의 값을 부여
- 여기서 **`mines`**는 1차원 배열, 배열의 각 값은 보드의 특정 셀에 대응. 이 값은 2차원 보드의 인덱스로 변환되어야 함.
- 이러한 변환을 위해 **`divmod`**를 사용. **`mine`** 값을 **`self.board_size`**로 나누어 몫(**`x`**)과 나머지(**`y`**)를 얻으면, 이를 통해 1차원 인덱스를 2차원 인덱스로 쉽게 변환 가능
- **`mine = 7`**인 경우, **`divmod(7, 5)`**는 **`(1, 2)`**를 반환합니다. 이는 (1, 2) 위치에 지뢰가 있음을 의미합니다. (Example)
- `for` 루프: 선택된 지뢰 위치를 보드에 배치합니다. `divmod` 함수를 사용하여 1차원 인덱스를 2차원 인덱스로 변환합니다. 지뢰가 있는 칸은 `-1`로 표시합니다.
- 두 개의 중첩된 `for` 루프를 사용하여 보드의 모든 칸을 순회합니다.
- 현재 칸이 지뢰인 경우(`1`), 인접한 지뢰 개수를 계산할 필요가 없으므로 `continue`로 건너뜁니다.
- `count`: 현재 칸 주변의 8개 칸을 검사하여 인접한 지뢰의 개수를 셉니다.
- 각 칸의 값을 인접한 지뢰의 개수로 설정합니다.

**초기 상태 설정**

- `self.done`: 게임 종료 여부를 나타내는 플래그를 `False`로 초기화합니다.
- `self.steps`: 현재 에피소드에서 수행한 단계 수를 0으로 초기화합니다.
- `self.total_reward`: 에피소드 동안 누적된 보상을 0으로 초기화합니다.
- `self.mine_hit`: 지뢰를 밟았는지 여부를 나타내는 플래그를 `False`로 초기화합니다.
- `self.first_click`: 첫 클릭 여부를 나타내는 플래그를 `True`로 설정합니다. 첫 클릭 시 지뢰를 재배치할 필요가 있기 때문입니다.

**초기 관측 반환**

- `_get_observation` 메서드를 호출하여 현재 보드 상태를 반환합니다. 이 상태는 에이전트가 다음 행동을 결정하기 위해 사용하는 입력으로 사용됩니다.

### Step

```python
def step(self, action):
        x, y = divmod(action, self.board_size)
        # unopened_cells = np.sum(self.state[:, :, 0] == 0)
        if self.first_click:
            # 첫 클릭 시 지뢰가 있는 경우 처리
            if self.board[x, y] == -1:
                self._relocate_mine(x, y)
            self.first_click = False

        if self.state[0, x, y] == 1:
            reward = -0.1  # 이미 열린 칸을 누르면 벌점
            done = False
        elif self.board[x, y] == -1:
            self.state[0, x, y] = 1
            reward = -10  # 지뢰를 누르면 큰 벌점
            self.mine_hit = True
            done = True
        else:
            self.reveal_cells(x, y)  # 주변 0인 칸을 여는 메서드 호출
            reward = 1  # 일반 칸을 누르면 보상
            done = self.check_done()

            if done and not self.mine_hit:
                reward = 10  # 게임을 클리어하면 큰 보상

        # 누적 보상 계산
        self.total_reward += reward
        self.steps += 1

        return self._get_observation(), reward, done, {}
```

**행동을 좌표로 변환**

- `divmod` 함수를 사용하여 1차원 인덱스를 2차원 좌표 `(x, y)`로 변환합니다. `action`은 보드의 특정 칸을 나타내는 1차원 인덱스입니다.

**첫 클릭 처리**

- `if self.first_click`: 첫 클릭인 경우, 해당 위치에 지뢰가 있는지 확인합니다.
- 지뢰가 있는 경우, `_relocate_mine` 메서드를 호출하여 지뢰를 다른 위치로 재배치합니다.
- 첫 클릭 이후에는 `self.first_click` 플래그를 `False`로 설정합니다.

**이미 열린 칸 클릭 처리**

- `if self.state[0, x, y] == 1`: 해당 칸이 이미 열려 있는 경우, 작은 벌점(`0.1`)을 부여하고 게임이 끝나지 않았음을 표시합니다(`done = False`).

**지뢰 클릭 처리**

- `elif self.board[x, y] == -1`: 해당 칸에 지뢰가 있는 경우, 그 칸을 열고 큰 벌점(`10`)을 부여합니다.
- `self.mine_hit` 플래그를 `True`로 설정하고, 게임이 종료되었음을 표시합니다(`done = True`).

**일반 칸 클릭 처리**

- `else`: 해당 칸이 지뢰가 아닌 경우, `reveal_cells` 메서드를 호출하여 해당 칸과 주변의 0인 칸을 엽니다.
- 보상(`1`)을 부여하고, 게임 종료 여부를 확인합니다(`done = self.check_done()`).
- 게임이 종료되었고 지뢰를 밟지 않은 경우, 큰 보상(`10`)을 추가로 부여합니다.

**누적 보상 및 단계 수 업데이트**

- `self.total_reward`에 현재 단계에서 얻은 보상을 더합니다.
- `self.steps` 값을 1 증가시켜 총 단계를 업데이트합니다.

**결과 반환**

- 현재 상태 관찰값(`_get_observation`), 보상(`reward`), 종료 여부(`done`), 추가 정보(`{}`)를 반환합니다.

### Relocate Mine

```python
def _relocate_mine(self, x, y):
        self.board[x, y] = 0
        possible_positions = set(range(self.board_size * self.board_size)) - set(self.mines)
        # 클릭 위치가 possible_positions에 있는지 확인하고 제거
        if x * self.board_size + y in possible_positions:
            possible_positions.remove(x * self.board_size + y)
        new_mine_position = np.random.choice(list(possible_positions))
        new_x, new_y = divmod(new_mine_position, self.board_size)
        self.board[new_x, new_y] = -1
        self.mines = [m for m in self.mines if m != x * self.board_size + y]
        self.mines.append(new_mine_position)
        self._update_board_counts()
```

**지뢰 제거**

- 지뢰가 있던 위치 `(x, y)`의 값을 0으로 설정하여 지뢰를 제거합니다.

**가능한 위치 계산**

- 보드의 모든 위치(0부터 `board_size * board_size`까지)의 집합에서 현재 지뢰가 있는 위치들의 집합을 뺍니다. 이를 통해 지뢰가 없는 가능한 위치들을 계산합니다.

**클릭 위치 제거**

- 클릭한 위치 `(x * self.board_size + y)`가 가능한 위치들에 포함되어 있는지 확인하고, 포함되어 있다면 제거합니다. 첫 클릭 위치에 지뢰가 다시 배치되지 않도록 하기 위함입니다.

**새 지뢰 위치 선택**

- 가능한 위치들 중 하나를 무작위로 선택하여 새로운 지뢰 위치로 설정합니다.

**새 지뢰 위치 좌표 변환**

- `divmod` 함수를 사용하여 새로운 지뢰 위치를 1차원 인덱스에서 2차원 좌표 `(new_x, new_y)`로 변환합니다.

**새 지뢰 위치 설정**

- 새로 선택된 위치 `(new_x, new_y)`에 지뢰를 배치합니다. 해당 위치의 값을 `1`로 설정합니다.

**지뢰 목록 업데이트**

- 기존 지뢰 목록에서 제거된 지뢰 위치를 제외하고, 새로운 지뢰 위치를 추가합니다. 이를 통해 지뢰 목록을 최신 상태로 유지합니다.

**보드 카운트 업데이트**

- `_update_board_counts` 메서드를 호출하여 보드의 각 칸에 인접한 지뢰의 개수를 다시 계산합니다. 지뢰 위치가 변경되었기 때문에 보드 카운트를 업데이트해야 합니다.

### update_board_counts

```python
def _update_board_counts(self):
        # 인접한 지뢰 개수 다시 계산
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == -1:
                    continue
                count = 0
                for x in range(max(0, i - 1), min(self.board_size, i + 2)):
                    for y in range(max(0, j - 1), min(self.board_size, j + 2)):
                        if self.board[x, y] == -1:
                            count += 1
                self.board[i, j] = count
```

**보드의 각 칸을 순회**

- 중첩된 `for` 루프를 사용하여 보드의 모든 칸을 순회합니다. `i`는 행 인덱스, `j`는 열 인덱스를 나타냅니다.

**지뢰인 칸은 건너뜀**

- 현재 칸이 지뢰인 경우(`self.board[i, j] == -1`), 그 칸은 인접한 지뢰 개수를 계산할 필요가 없으므로 `continue`를 사용하여 다음 칸으로 건너뜁니다.

**인접 칸을 순회**

- 중첩된 `for` 루프를 사용하여 현재 칸 `(i, j)`에 인접한 모든 칸을 순회합니다.
- `range(max(0, i - 1), min(self.board_size, i + 2))`는 현재 칸의 위, 아래 인접한 행을 선택합니다.
- `range(max(0, j - 1), min(self.board_size, j + 2))`는 현재 칸의 왼쪽, 오른쪽 인접한 열을 선택합니다.
- `max`와 `min` 함수를 사용하여 보드 경계를 벗어나지 않도록 합니다.

**지뢰 개수 세기**

- 인접한 칸이 지뢰인 경우(`self.board[x, y] == -1`), `count`를 1 증가시켜 지뢰 개수를 셉니다.

**카운트 저장**

- 현재 칸 `(i, j)`에 인접한 지뢰의 개수를 계산한 후, 그 값을 `self.board[i, j]`에 저장합니다.

### Reveal_Cells

```python
# 주변 0인 칸을 여는 메서드
    def reveal_cells(self, x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cx >= self.board_size or cy < 0 or cy >= self.board_size:
                continue
            if self.state[0, cx, cy] == 1:
                continue

            self.state[0, cx, cy] = 1
            self.state[1, cx, cy] = self.board[cx, cy]

            if self.board[cx, cy] == 0:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if dx != 0 or dy != 0:
                            stack.append((cx + dx, cy + dy))
```

**메서드 선언 및 초기화**

- `reveal_cells` 메서드는 시작 좌표 `(x, y)`를 입력으로 받아 실행됩니다.
- `stack` 리스트를 초기화하고, 시작 좌표 `(x, y)`를 스택에 추가합니다. 스택은 열어야 할 칸들을 추적하는 데 사용됩니다.

**스택 순회**

- 스택이 비어 있지 않은 동안 반복문을 실행합니다.
- 스택의 마지막 요소를 꺼내 현재 좌표 `(cx, cy)`로 설정합니다.

**보드 경계 체크**

- 현재 좌표 `(cx, cy)`가 보드 경계를 벗어나는지 확인합니다. 경계를 벗어난 경우, 다음 반복으로 건너뜁니다.

**이미 열린 칸 체크**

- 현재 칸이 이미 열려 있는지 확인합니다(`self.state[0, cx, cy] == 1`). 이미 열려 있다면, 다음 반복으로 건너뜁니다.

**칸 열기**

- 현재 칸을 열고(`self.state[0, cx, cy] = 1`), 현재 칸의 값을 상태에 업데이트합니다(`self.state[1, cx, cy] = self.board[cx, cy]`).

**주변 칸 열기**

- 현재 칸의 값이 0인 경우, 인접한 모든 칸을 확인합니다.
- `for` 루프를 사용하여 현재 칸의 주변 8개 칸을 순회합니다(`dx`와 `dy`는 -1부터 1까지의 범위를 가집니다).
- 현재 칸 자체는 제외하기 위해 `if dx != 0 or dy != 0` 조건을 사용합니다.
- 인접한 칸의 좌표 `(cx + dx, cy + dy)`를 스택에 추가합니다.

### 상태 정규화

```python
def _get_observation(self):
        # 상태 정규화
        norm_state = self.state.astype(np.float32)
        norm_state[1, :, :] = norm_state[1, :, :] / 8.0  # 주변 지뢰 개수를 0-1 사이로 정규화
        return norm_state
```

**상태 정규화 준비**

- `self.state` 배열을 `np.float32` 타입으로 변환하여 `norm_state`에 저장합니다. 이는 정규화를 수행하기 위해 데이터 타입을 부동 소수점 형식으로 변경하는 과정입니다.

**주변 지뢰 개수 정규화**

- `norm_state`의 두 번째 채널(`norm_state[1, :, :]`)에 대해 모든 값을 8.0으로 나누어 정규화합니다.
- Minesweeper 보드의 각 칸은 최대 8개의 인접 지뢰를 가질 수 있으므로, 이를 0-1 범위로 정규화하여 신경망의 입력으로 사용하기 쉽게 만듭니다.

### 게임 클리어 여부

```python
def check_done(self):
        unopened_cells = np.sum(self.state[0, :, :] == 0)
        if unopened_cells == self.num_mines:
            return True
        return False
```

### 렌더링 예시

- **게임 오버**
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/33f7fb8b-4aed-4519-9dbe-40e79ad2b724/853ea19c-de4e-471c-82d2-136cd722a20c/Untitled.png)
    

- **클리어**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/33f7fb8b-4aed-4519-9dbe-40e79ad2b724/7ad40bf8-b1ff-42bc-ab86-f1e8b542110a/Untitled.png)

---

# 에이전트 (이은나)

## CNN 신경망

```python
class CNN(nn.Module):
    def __init__(self, state_shape, action_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3), stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  

        conv_output_size = self._get_conv_output(state_shape)
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape)
        output = self.pool1(torch.relu(self.conv1(input)))
        output = self.pool2(torch.relu(self.conv2(output)))  
        return int(np.prod(output.size()))

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x))) 
        x = self.pool2(torch.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

---

```python
class CNN(nn.Module):
    def __init__(self, state_shape, action_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 3), stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  

        conv_output_size = self._get_conv_output(state_shape)
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, action_size)
```

- 클래스 정의
    - state_shape: 입력 상태의 형태, (2, 9, 9)
    - action_size: 가능한 행동의 수, 9X9=81
    - self.conv1: 첫 번째 컨볼루션 레이어
        - input (2, 9, 9) → output (16, 11, 11)
    - self.pool1: 풀링 레이어1
        - input(16, 11, 11) → output (16, 6, 6)
    - self.conv2: 두 번째 컨볼루션 레이어
        - input (16, 6, 6) → output (32, 6, 6)
    - self.poo2l: 풀링 레이어2
        - input (32, 6, 6) → output (32, 3, 3)
    - flatten : 1차원 벡터로 변환
        - 32x3x3 = 288
    - self.fc1: 첫 번째 완전 연결층
        - input 288 → output 256
    - self.fc2: 두 번째 완전 연결층
        - input 256 → output 81
    

```python
def _get_conv_output(self, shape):
     input = torch.rand(1, *shape)
     output = self.pool1(torch.relu(self.conv1(input)))
     output = self.pool2(torch.relu(self.conv2(output)))  
     return int(np.prod(output.size()))
```

- _get_conv_output : 컨볼루션 레이어와 풀링 레이어를 통과한 후의 출력 크기를 계산
    - shape : 입력 상태의 형태를 나타내는 튜플, (2, 9, 9)
    - input : (1, 2, 9, 9)

```python
def forward(self, x):
     x = self.pool1(torch.relu(self.conv1(x))) 
     x = self.pool2(torch.relu(self.conv2(x))) 
     x = x.view(x.size(0), -1)
     x = torch.relu(self.fc1(x))
     x = self.fc2(x)
     return x
```

- forward : 신경망의 순전파를 정의, 신경망을 순차적으로 통과시키면서 출력을 계산

## 하이퍼파라미터

```python
MEM_SIZE_MAX = 50000 # 최대 메모리 크기
MEM_SIZE_MIN = 1000 # 최소 메모리 크기
BATCH_SIZE = 64 # 배치 크기

LEARNING_RATE = 0.001 # 학습률
DISCOUNT = 0.1 # 할인율(gamma)

EPSILON = 1.0 # 초기 탐색률
EPSILON_DECAY = 0.995 # 탐색률 감소율
EPSILON_MIN = 0.01 # 최소 탐색률

UPDATE_TARGET_EVERY = 5 # 타겟 네트워크 업데이트 주기
EPISODES = 50000 # 총 에피소드 수
MAX_STEPS = 71 # 에피소드 내 최대 스텝 수
```

## DQN 에이전트

```python
class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=MEM_SIZE_MAX)
        
        self.model = CNN(state_shape, action_size).cuda()
        self.target_model = CNN(state_shape, action_size).cuda()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        
        self.epsilon = EPSILON
        self.losses = [] 

    def update_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = self.normalize_state(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).cuda()
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < MEM_SIZE_MIN:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)

        states = torch.FloatTensor([self.normalize_state(m[0]) for m in minibatch]).cuda()
        actions = torch.LongTensor([m[1] for m in minibatch]).cuda()
        rewards = torch.FloatTensor([m[2] for m in minibatch]).cuda()
        next_states = torch.FloatTensor([self.normalize_state(m[3]) for m in minibatch]).cuda()
        dones = torch.FloatTensor([m[4] for m in minibatch]).cuda()

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (DISCOUNT * next_q_values * (1 - dones))

        loss = self.loss_fn(q_values, target_q_values.detach())
        self.losses.append(loss.item()) 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def normalize_state(self, state):
        norm_state = state.astype(np.float32)
        norm_state[1, :, :] = norm_state[1, :, :] / 8.0  
        return norm_state
```

---

```python
def __init__(self, state_shape, action_size):
    self.state_shape = state_shape
    self.action_size = action_size
    self.memory = deque(maxlen=MEM_SIZE_MAX)
    
    self.model = CNN(state_shape, action_size).cuda()
    self.target_model = CNN(state_shape, action_size).cuda()
    self.target_model.load_state_dict(self.model.state_dict())
    self.target_model.eval()

    self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
    self.loss_fn = nn.MSELoss()
    
    self.epsilon = EPSILON
    self.losses = []
```

- 초기화 메서드 (__init__)
    - **state_shape**: 상태의 형태
    - **action_size**: 가능한 행동의 수
    - **memory**: 경험을 저장할 덱(큐)
    - **model**: 현재 Q-네트워크
    - **target_model**: 타겟 Q-네트워크
    - **optimizer**: Adam 옵티마이저
    - **loss_fn**: 손실 함수(MSE)
    - **epsilon**: 탐색율
    - **losses**: 학습 동안 손실을 기록할 리스트

```python
def update_epsilon(self):
    if self.epsilon > EPSILON_MIN:
        self.epsilon *= EPSILON_DECAY
```

- update_epsilon : 탐색율 업데이트
    - 탐색율을 줄여나가는 역할

```python
def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
```

- remember : 상태, 행동, 보상, 다음 상태, 완료 여부를 메모리에 저장

```python
def act(self, state):
    state = self.normalize_state(state)
    if np.random.rand() <= self.epsilon:
        return random.randrange(self.action_size)
    state = torch.FloatTensor(state).unsqueeze(0).cuda()
    with torch.no_grad():
        act_values = self.model(state)
    return torch.argmax(act_values).item()
```

- act : 현재 상태에서 행동을 선택
    - 탐색율에 따라 무작위 행동 또는 모델 예측에 기반한 행동을 선택

```python
def replay(self):
    if len(self.memory) < MEM_SIZE_MIN:
        return

    minibatch = random.sample(self.memory, BATCH_SIZE)

    states = torch.FloatTensor([self.normalize_state(m[0]) for m in minibatch]).cuda()
    actions = torch.LongTensor([m[1] for m in minibatch]).cuda()
    rewards = torch.FloatTensor([m[2] for m in minibatch]).cuda()
    next_states = torch.FloatTensor([self.normalize_state(m[3]) for m in minibatch]).cuda()
    dones = torch.FloatTensor([m[4] for m in minibatch]).cuda()

    q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = self.target_model(next_states).max(1)[0]
    target_q_values = rewards + (DISCOUNT * next_q_values * (1 - dones))

    loss = self.loss_fn(q_values, target_q_values.detach())
    self.losses.append(loss.item()) 
    
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

- replay : 메모리에 저장된 경험으로부터 샘플을 추출하여 모델을 학습
    
    ```python
    if len(self.memory) < MEM_SIZE_MIN:
    		return
    ```
    
    - 메모리의 크기가 MEM_SIZE_MIN보다 작은 경우, 메서드를 종료
    
    ```python
    minibatch = random.sample(self.memory, BATCH_SIZE)
    
    ```
    
    - 미니배치 샘플링
    
    ```python
    states = torch.FloatTensor([self.normalize_state(m[0]) for m in minibatch]).cuda()
    actions = torch.LongTensor([m[1] for m in minibatch]).cuda()
    rewards = torch.FloatTensor([m[2] for m in minibatch]).cuda()
    next_states = torch.FloatTensor([self.normalize_state(m[3]) for m in minibatch]).cuda()
    dones = torch.FloatTensor([m[4] for m in minibatch]).cuda()
    ```
    
    - 샘플링된 미니배치의 각 요소를 PyTorch 텐서로 변환
    - GPU로 이동
    - self.normalize_state : 상태 정규화 함수
        - states, next_states에 적용
    
    ```python
    q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = self.target_model(next_states).max(1)[0]
    target_q_values = rewards + (DISCOUNT * next_q_values * (1 - dones))
    ```
    
    - 현재 상태에 대한 Q-값 계산
    - 다음 상태에 대한 최대 Q-값 계산
    - 타겟 Q-값 계산
    
    ```python
    loss = self.loss_fn(q_values, target_q_values.detach())
    self.losses.append(loss.item())
    ```
    
    - 손실 계산 : mse (예측 Q-값과 타겟 Q-값 간의 차이를 계산)
    - 손실 기록
    
    ```python
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    ```
    
    - 기울기 초기화
    - 역전파 : 손실 값을 기준으로 신경망의 가중치에 대한 기울기를 계산
    - 신경망 가중치 업데이트

```python
def update_target_model(self):
    self.target_model.load_state_dict(self.model.state_dict())
```

- update_target_model : 타겟 모델의 가중치를 현재 모델의 가중치로 업데이트

```python
def normalize_state(self, state):
    norm_state = state.astype(np.float32)
    norm_state[1, :, :] = norm_state[1, :, :] / 8.0
    return norm_state
```

- normalize_state  : 상태 정규화 함수
    - 두 번째 채널(주변 지뢰 개수)을 0-1 사이로 정규화
    - 최대 지뢰의 수 8개

## 시각화 함수

```python
def plot_metrics(episode_list, avg_rewards, avg_steps, success_rates, avg_loss):
    plt.figure(figsize=(20, 12))

    plt.subplot(2, 2, 1)
    plt.plot(episode_list, avg_rewards, label='Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward over Episodes')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(episode_list, avg_steps, label='Average Steps', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps')
    plt.title('Average Steps over Episodes')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(episode_list, success_rates, label='Success Rate', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate over Episodes')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(episode_list, avg_loss, label='Average Loss', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Loss')
    plt.title('Average Loss over Episodes')
    plt.legend()

    plt.tight_layout()
    plt.show()
```

- 주요 인수
    - episode_list: 각 에피소드 번호 리스트
    - avg_rewards: 100 에피소드별 평균 보상 리스트
    - avg_steps: 100 에피소드별 평균 스텝 수 리스트
    - success_rates: 100 에피소드별 성공률 리스트
    - avg_loss: 100 에피소드별 평균 손실 리스트
- 그래프
    - 평균 보상 그래프
    - 평균 스텝 수 그래프
    - 성공률 그래프
    - 평균 손실 그래프

## 훈련

```python
env = MinesweeperEnv()
state_shape = env.observation_space.shape
action_size = env.action_space.n

agent = DQNAgent(state_shape, action_size)

episode_rewards = [] 
total_steps = []  
success_rates = [] 
episode_list = []
losses = []  
avg_rewards = []  
avg_steps = []  
avg_loss = []
success_count = 0

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0  
    while not done and steps < MAX_STEPS: 
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        agent.replay()
        steps += 1  

    agent.update_epsilon()
    total_steps.append(steps) 
    episode_rewards.append(total_reward)  

    if episode % UPDATE_TARGET_EVERY == 0:
        agent.update_target_model()

    if done and (reward == 10 or not env.mine_hit): 
        success_count += 1

    if (episode + 1) % 100 == 0:
        avg_reward_last_100 = np.mean(episode_rewards[-100:])
        avg_steps_last_100 = np.mean(total_steps[-100:]) 
        success_rate = (success_count / (episode + 1)) * 100
        avg_loss_last_100 = np.mean(agent.losses[-100:])
        
        avg_rewards.append(avg_reward_last_100)
        avg_steps.append(avg_steps_last_100)
        success_rates.append(success_rate)
        episode_list.append(episode + 1)
        avg_loss.append(avg_loss_last_100)
        
        print(f"Episode: {episode + 1}, Average Reward : {avg_reward_last_100:.2f}, Average Steps: {avg_steps_last_100:.2f}, Success Rate: {success_rate:.2f}%, Avg Loss: {avg_loss_last_100:.4f}, Epsilon: {agent.epsilon:.4f}")

plot_metrics(episode_list, avg_rewards, avg_steps, success_rates, avg_loss)
```

---

```python
episode_rewards = []  # 각 에피소드의 총 보상
total_steps = []  # 각 에피소드의 총 스텝 수
success_rates = []  # 성공률
episode_list = []  # 에피소드 번호
losses = []  # 손실 기록
avg_rewards = []  # 평균 보상
avg_steps = []  # 평균 스텝 수
avg_loss = []  # 평균 손실
success_count = 0  # 성공 횟수
```

- 변수 초기화

```python
while not done and steps < MAX_STEPS: 
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    total_reward += reward

    agent.remember(state, action, reward, next_state, done)
    state = next_state

    agent.replay()
    steps += 1  
```

- 최대 스텝 수 지정
    - MAX_STEPS = 71 (총 칸 수 81 - 지뢰 10)
    - 에피소드가 완료되지 않았거나 최대 스텝 수에 도달하지 않았을 때까지 반복

```python
if done and (reward == 10 or not env.mine_hit): 
    success_count += 1
```

- 성공 조건
    - done: 현재 에피소드가 완료
    - reward == 10 : 마지막 행동에서 받은 보상
        
        / not env.mine_hit : 지뢰를 밟지 않았을 때 참
        

```python
avg_reward_last_100 = np.mean(episode_rewards[-100:])
avg_steps_last_100 = np.mean(total_steps[-100:])
success_rate = (success_count / (episode + 1)) * 100
avg_loss_last_100 = np.mean(agent.losses[-100:])
```

- 학습 성과 지표
    - avg_reward_last_100: 마지막 100개 에피소드에서의 평균 보상
    - avg_steps_last_100: 마지막 100개 에피소드에서의 평균 스텝 수
    - success_rate: 전체 에피소드 중 성공적인 에피소드의 비율
    - avg_loss_last_100: 마지막 100개 학습 단계에서의 평균 손실

```python
avg_rewards.append(avg_reward_last_100)
avg_steps.append(avg_steps_last_100)
success_rates.append(success_rate)
episode_list.append(episode + 1)
avg_loss.append(avg_loss_last_100)
```

- 학습 성과 지표를 리스트에 추가
    - avg_rewards: 리스트에 최근 100개 에피소드의 평균 보상을 추가
    - avg_steps: 리스트에 최근 100개 에피소드의 평균 스텝 수를 추가
    - success_rates: 리스트에 현재까지의 성공률을 추가
    - episode_list: 리스트에 현재 에피소드 번호를 추가
    - avg_loss: 리스트에 최근 100개 학습 단계의 평균 손실을 추가

## 테스트

```python
agent.epsilon = 0.0 
success_count = 0

episodes = 5000

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done and steps < MAX_STEPS:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

    success = done and (reward == 10 or not env.mine_hit)
    if success:
        success_count += 1

    print(f"Episode {episode + 1}: Success: {success}, Steps: {steps}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

success_rate = (success_count / episodes) * 100
print(f"Success Rate: {success_rate:.2f}%")
```

- epsilon 0
- Success Rate 출력

---

### Train 1

에피소드 50000번

<aside>
❗ - 배치사이즈 32→64
- gamma 0.99 → 0.1
-에이전트에 상태 정규화 추가
- 훈련 max step 수를 71로 제한

</aside>

1. **Average Reward** : 초반 에피소드에서 평균 보상은 약 -6.0에서 -7.0, 중반 이후로 평균 보상은 -5.5에서 -6.5 사이에서 변동합니다. 평균 보상이 전반적으로 개선되지 않고, 음수 상태이므로 지뢰를 피하지 못하고 자주 밟고 있다는 것을 알 수 있습니다.
2. **Average Steps** : 초기 에피소드에서 평균 스텝 수는 약 10에서 시작하여 중반 이후로 평균 스텝 수는 약 40에서 50 사이에서 변동합니다. 최대 스텝 수에 가까워지며 학습이 잘되지 않는 경향을 보입니다.
3. **Success Rate** : 약 만 번 에피소드가 돌아갔을 때 0.008%로 성공률이 올랐지만, 이후 거의 0에 수렴합니다.
4. **Average Loss** : 초기 에피소드 손실 값은 약 8.0에서 중반 이후 손실 값은 빠르게 감소하여 1.0 이하로 유지됩니다.

### Test1

- 5000번 테스트 → Success Rate: 0.00%
    
    

---

### Train 2

에피소드 10000번

<aside>
❗ - 리워드 수정 : 이미 열린 칸을 누르면 벌점 -0.1 → -1
- MEM_SIZE_MAX 100000 → 50000 
- UPDATE_TARGET_EVERY 10 → 5

</aside>

1. **Average Reward** : 초반 에피소드에서 보상은 약 -10, 이후로 평균 보상은 -30에서 -45 사이에서 변동합니다. 평균 보상이 전반적으로 개선되지 않고, 음수 상태이므로 지뢰를 피하지 못하고 자주 밟고 있다는 것을 알 수 있습니다.
2. **Average Steps** : 초기 에피소드에서 평균 스텝 수는 약 10에서 시작하여 중반 이후로 평균 스텝 수는 약 40에서 50 사이에서 변동합니다. 최대 스텝 수에 가까워지며 학습이 잘되지 않는 경향을 보입니다.
3. **Success Rate** : 약 천 번 에피소드가 돌아갔을 때 0.35%로 성공률이 올랐지만, 이후 거의 0에 수렴합니다.
4. **Average Loss** : 초기 에피소드 손실 값은 약 10에서 중반 이후 손실 값은 빠르게 감소하여 1.0 이하로 유지됩니다.

### Test2

- 1000번 테스트 → Success Rate: 0.00%
    

⇒ 테스트 시 스텝 수가 71로 실패하거나  한 자릿수에서 지뢰를 누른 경우가 많았습니다. 훈련 시 학습이 제대로 되지 않은 것 같습니다. 

⇒ 최소 스텝 수를 제한, 하이퍼파라미터 조정, 신경망 변경 등..
