# 큰 이미지에서 자동 손 탐지
<br/>
작은 이미지의 경우 mediapipe 라이브러리를 사용하면 쉽게 손을 탐지할 수 있지만, 큰 이미지의 경우 인식률이 떨어지는 경우가 발생합니다.
<br/>
이를 해결하기 위해, 사람 단위로 사진을 잘라서 손을 탐지하는 과정을 만들어봤습니다.
<br/>
<br/>
먼저 Yolo를 이용해서 사람이 있는 영역을 찾아냅니다.
<br/>
사람이 있는 영역을 여유를 더 두고 잘라냅니다.
<br/>
그 영역에서 mediapipe로 손을 탐지합니다.
<br/>
<br/>
<h1> Automatic Hand Detection In Large Images </h1>
<br/>
For small images, the mediapipe library can easily detect hands, but for large images, the recognition rate may be low.
<br/>
To solve this problem, I created a process to detect hands by cutting pictures in units of people.
<br/>
<br/>
First, use Yolo to find areas where the person is.
<br/>
Next, cut the areas where the person is, leaving more space.
<br/>
Finally, detect hands with mediapipe in that areas.
<br/>
