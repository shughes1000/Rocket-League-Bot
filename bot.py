from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time, predict_future_goal
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target, pitch_toward_target, yaw_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3
from util.orientation import Orientation, relative_location

import math


class MyBot(BaseAgent):

    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())
        
        # blue net: negative y
        self.info = self.get_field_info()
        
        self.my_goal_location = Vec3(self.info.goals[self.team].location)
        
        if self.team == 0:
            self.team_coef = -1
            self.opp_goal_location = Vec3(self.info.goals[1].location)
            self.opp_goal_left_post = Vec3(-800, 5120, 0)
            self.opp_goal_right_post = Vec3(800, 5120, 0)
            self.my_goal_left_post = Vec3(-800, -5120, 0)
            self.my_goal_right_post = Vec3(800, -5120, 0)
        else:
            self.team_coef = 1
            self.opp_goal_location = Vec3(self.info.goals[0].location)
            self.opp_goal_left_post = Vec3(-800, -5120, 0)
            self.opp_goal_right_post = Vec3(800, -5120, 0)
            self.my_goal_left_post = Vec3(-800, 5120, 0)
            self.my_goal_right_post = Vec3(800, 5120, 0)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """
        
        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # return
            
        ### DECLARATIONS
        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)
        ball_rotation = packet.game_ball.physics.rotation
        ball_orientation = Orientation(ball_rotation)
        ball_velocity = Vec3(packet.game_ball.physics.velocity)
        
        car_rotation = my_car.physics.rotation
        car_orientation = Orientation(car_rotation)
        car_to_ball = relative_location(car_location, car_orientation, ball_location)
        
        car_roll = car_rotation.roll
        car_pitch = car_rotation.pitch
        car_pitch_in_degrees = car_pitch * 90
        car_yaw = car_rotation.yaw
        
        target_location = ball_location
        
        # my_goal_left_post = self.my_goal_left_post
        # my_goal_right_post = self.my_goal_right_post
        
        
        car_on_wheels = False
        if my_car.has_wheel_contact == True:
            car_on_wheels = True
            
        car_grounded = False
        if car_location[2] <= 17.1:
            car_grounded = True
        
        ball_grounded = False
        if ball_location[2] <= 150:
            ball_grounded = True
            
        ball_in_air = False
        if ball_location[2] > 500:
            ball_in_air = True
        
        # blue net: negative y
        info = self.info
        
        # blue is 0, orange is 1
        color = self.team
        
        opp_car = None
        opp_car_location = Vec3(0, 0, 999999)
        for i in range(packet.num_cars):
            if i == self.index:
                continue
            current_opp = packet.game_cars[i]
            if current_opp.team == color:
                continue
            if Vec3(current_opp.physics.location).dist(ball_location) < opp_car_location.dist(ball_location):
                opp_car = current_opp
                opp_car_location = Vec3(opp_car.physics.location)
        

        if car_location.x > 0:
            defense_location = Vec3(700, self.team_coef * 5000, 0)
        else:
            defense_location = Vec3(-700, self.team_coef * 5000, 0)
        
        nearest_big_boost = Vec3(get_nearest_big_boost(info, packet, car_location, car_orientation))
            
        nearest_small_boost = Vec3(get_nearest_small_boost(info, packet, car_location, car_orientation))


        ### BALL PREDICTION
        ball_prediction = self.get_ball_prediction_struct()  # This can predict bounces, etc
        if car_velocity.length() == 0:
            ball_in_future = find_slice_at_time(ball_prediction,
                                                packet.game_info.seconds_elapsed + car_location.dist(ball_location))
        else:
            ball_in_future = find_slice_at_time(ball_prediction,
                                                packet.game_info.seconds_elapsed + 2)
        
        if car_location.dist(ball_location) > 5000:
            distance = 5000
        else:
            distance = car_location.dist(ball_location)
        if ball_in_future:
            ball_path = (distance/5000) * Vec3(ball_in_future.physics.location) + (1-distance/5000) * ball_location
        else:
            ball_path = ball_location
            
        potential_goal = False
        potential_goal_slice = predict_future_goal(ball_prediction)
        if color == 0:
            if potential_goal_slice and ball_path.y < 0:
                potential_goal = True
        else:
            if potential_goal_slice and ball_path.y > 0:
                potential_goal = True

        ball_path_grounded = Vec3(ball_path.x, ball_path.y, 0)
        
        # ball_path_to_goal = Vec3(1.5*(ball_path.x - self.opp_goal_location.x), ball_path.y + self.team_coef*abs(1.5*(ball_path.x - self.opp_goal_location.x)), 0)

        # if car_location.dist(ball_path_to_goal) < 100 or car_location.dist(ball_path) < car_location.dist(ball_path_to_goal):
        #     ball_path_to_goal = ball_path_grounded
        
        car_angle_to_left, car_angle_to_right, get_positioning_on_ball, shooting_angle = triangle(self.opp_goal_location, self.opp_goal_left_post, self.opp_goal_right_post, ball_path, ball_velocity, car_location)
        car_angle_to_left_own, car_angle_to_right_own, dont_own_goal, own_goal_angle = triangle(self.my_goal_location, self.my_goal_left_post, self.my_goal_right_post, ball_path, ball_velocity, car_location)
        # triangle_length = self.opp_goal_location.y - ball_path.y
        # triangle_width = self.opp_goal_location.x - ball_path.x
        # triangle_hyp = (triangle_length**2 + triangle_width**2)**(1/2)
        # ball_path_angle_to_net = math.atan2(triangle_length, triangle_width)
        # new_hyp = triangle_hyp*(1 + ball_velocity.length()/15000)
        # new_y = new_hyp * math.sin(ball_path_angle_to_net)
        # new_x = new_hyp * math.cos(ball_path_angle_to_net)
        
        # triangle_car_length = self.opp_goal_location.y - car_location.y
        # triangle_width_left = self.opp_goal_left_post.x - car_location.x
        # car_angle_to_left = math.atan2(triangle_car_length, triangle_width_left)
        # triangle_width_right = self.opp_goal_right_post.x - car_location.x
        # car_angle_to_right = math.atan2(triangle_car_length, triangle_width_right)
        
        
        # get_positioning_on_ball = Vec3(self.opp_goal_location.x - new_x, self.opp_goal_location.y - new_y, 0)
        # if get_positioning_on_ball.x > 5100:
        #     get_positioning_on_ball.x = 5100
        # elif get_positioning_on_ball.x < -5100:
        #     get_positioning_on_ball.x = -5100
        
        
        # shooting_angle = False
        # if ball_path.dist(self.opp_goal_location) < car_location.dist(self.opp_goal_location) and (car_angle_to_left < ball_path_angle_to_net < car_angle_to_right or car_angle_to_right < ball_path_angle_to_net < car_angle_to_left):
        #     shooting_angle = True
        
        
        
        ### SET BEHAVIOR
        behavior = "Ballchase"
        
        if car_location.dist(nearest_big_boost) < car_location.dist(ball_location) and my_car.boost < 50 and car_location.dist(self.my_goal_location) < ball_location.dist(self.my_goal_location) and ball_location.dist(self.my_goal_location) > nearest_big_boost.dist(self.my_goal_location):
            behavior = "Get big boost"
            
        if car_location.dist(nearest_small_boost) < 500 and my_car.boost < 75:
            behavior = "Get small boost"
        
        if opp_car_location.dist(ball_path) < car_location.dist(ball_path) and ball_path.dist(self.my_goal_location) < 2000:
            behavior = "Defense"
        
        if abs(defense_location[1] - ball_path[1]) < abs(defense_location[1] - car_location[1]):
            behavior = "Reposition"
        
        if abs(car_location[1]) > 5120:
            behavior = "Leave net"
        
        if potential_goal == True:
            behavior = "Save"
        
        self.has_first_touch_happened_yet = True
        if packet.game_info.is_kickoff_pause:
            self.has_first_touch_happened_yet = False
        if not self.has_first_touch_happened_yet:
            behavior = "Kickoff"
            # during kickoff, go for the ball
            target_location = ball_location
        
        # Don't chase the ball into offensive corners
        if behavior == "Ballchase" and abs(ball_location.x) > 900 and abs(ball_path.y) > 4900 and ball_path.dist(self.opp_goal_location) < ball_path.dist(self.my_goal_location):
            behavior = "Reposition"
            
        # if my_car.boost > 10:
        #     behavior = "Kill opp"
        
        # if behavior != "Reposition":
        #     behavior = "Ball chase"
        
        # behavior = "Jump at posts"
        # behavior = "Test"
        
        ### BEHAVIOR TO TARGET
        # TODO: Expand save so bot won't own goal (using relative location and angle of car to own net)
        if behavior == "Kickoff":
            target_location = ball_path
            self.renderer.draw_line_3d(ball_location, target_location, self.renderer.cyan())
        elif behavior == "Leave net":
            target_location = self.leave_net(car_location, color, ball_path)
        elif behavior == "Reposition":
            target_location = self.reposition(defense_location, car_location)
        elif behavior == "Get big boost":
            target_location = nearest_big_boost
        elif behavior == "Get small boost":
            target_location = nearest_small_boost
        elif behavior == "Ballchase":
            if shooting_angle: target_location = ball_path_grounded
            else: target_location = get_positioning_on_ball
            self.renderer.draw_line_3d(ball_location, target_location, self.renderer.cyan())
        elif behavior == "Attack":
            target_location = self.attack(ball_location, ball_path, car_location, self.opp_goal_location)
        elif behavior == "Defense":
            target_location = Vec3(ball_path.x, self.team_coef * 5100, 0)
        elif behavior == "Save":
            target_location = ball_path

        # elif behavior == "Kill opp":
        #     target_location = opp_car_location
        # elif behavior == "Jump at posts":
        #     num = round(packet.game_info.seconds_elapsed)
        #     num = num // 10
        #     if num % 2 == 1:
        #         target_location = Vec3(0, 5120, 1000)
        #     else:
        #         target_location = Vec3(0, -5120, 1000)

        
        car_to_target = relative_location(car_location, car_orientation, target_location)
        car_to_target_angle = math.atan2(car_to_target.y, car_to_target.x)*180/math.pi
        two_d_distance = (car_to_target.x**2 + car_to_target.y**2)**(1/2)
        car_to_target_verticle_angle = math.atan(abs(car_to_target.z/two_d_distance))*180/math.pi + car_pitch_in_degrees
        if car_to_target.z < 0:
            car_to_target_verticle_angle = -car_to_target_verticle_angle
        
        
        car_facing_target = False
        if abs(car_to_target_angle) <= 10:
            car_facing_target = True
        
        car_facing_aerial = False
        if abs(car_to_target_verticle_angle - car_pitch_in_degrees) <= 10:
            car_facing_aerial = True
        
        ### DEBUG
        # Draw some things to help understand what the bot is thinking
        car_debug = f'Speed: {car_velocity.length():.1f}\n'
        # car_debug += f"Location: {car_location}\n"
        # car_debug += f"Ball location: {ball_location}\n"
        car_debug += f"Behavior: {behavior}\n"
        # car_debug += f"Distance to target: {car_location.dist(target_location):.1f}\n"
        # car_debug += f"Verticle angle: {car_to_target_verticle_angle:.1f}\n"
        # car_debug += f"Ball speed: {ball_velocity.length():.1f}"
        # car_debug += f"Yaw: {car_yaw:.1f}\n"
        # car_debug += f"Pitch in degrees: {car_pitch_in_degrees:.1f}\n"
        # car_debug += f"Car grounded: {car_grounded}\n"
        # car_debug += f"Car facing target: {car_facing_target}\n"
        # car_debug += f"Car facing aerial: {car_facing_aerial}\n"
        # car_debug += f"x: {car_to_target.x:.1f}\n"
        # car_debug += f"y: {car_to_target.y:.1f}\n"
        # car_debug += f"z: {car_to_target.z:.1f}\n"
        # car_debug += f"2d: {two_d_distance:.1f}\n"
        # car_debug += f"Car angle to left post: {(car_angle_to_left)*180/math.pi:.1f}\n"
        # car_debug += f"Ball path angle to opp goal: {(ball_path_angle_to_net)*180/math.pi:.1f}\n"
        # car_debug += f"Car angle to right post: {(car_angle_to_right)*180/math.pi:.1f}\n"
        car_debug += f"Shooting Angle: {shooting_angle}\n"
        car_debug += f"Own Goal Angle: {own_goal_angle}\n"
        # car_debug += f"Potential Goal Condeded: {potential_goal}\n"
        
        
        self.renderer.draw_line_3d(car_location, self.opp_goal_left_post, self.renderer.red())
        self.renderer.draw_line_3d(car_location, self.opp_goal_right_post, self.renderer.red())
        # self.renderer.draw_line_3d(car_location, self.my_goal_left_post, self.renderer.orange())
        # self.renderer.draw_line_3d(car_location, self.my_goal_right_post, self.renderer.orange())
        self.debug(car_location, target_location, car_debug)
        
        
        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence is not None and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls
            
        ### CONTROLS/ACTIONS
        controls = SimpleControllerState()
        controls.steer = steer_toward_target(my_car, target_location)
        # controls.pitch = pitch_toward_target(my_car, target_location)
        # controls.yaw = yaw_toward_target(my_car, target_location)
        controls.throttle = 1.0
        # if not car_grounded or not car_on_wheels:
        #     controls.boost = True
        if car_location.dist(ball_path) < 1000 and target_location == get_positioning_on_ball:
            controls.boost = False
            controls.throttle = 0.5
        
        
        car_steering = True
        if (abs(controls.steer) < 0.3):
            car_steering = False
        
        
        ### BEHAVIORAL CONTROLS
        # Preset Kickoffs
        if behavior == "Kickoff":
            op_kickoff = self.op_kickoffs(packet, car_location, (-self.team_coef)*1)
            if op_kickoff:
                return op_kickoff
            kickoff_finish = self.kickoff_flip(packet, car_to_target_angle, car_location.dist(ball_location), controls, car_to_ball)
            if kickoff_finish:
                return kickoff_finish
        
        # TODO: Work on this
        if behavior == "Defense" and car_location.dist(target_location) < 100:
            controls.boost = False
            if car_velocity.length() > 300:
                controls.throttle = -0.3
            else:
                controls.throttle = 0.3
        
        # Stop Gunning it into own net
        if color == 0:
            if behavior == "Reposition" and car_location.y < -5000:
                controls.boost = False
                if car_velocity.length() > 300:
                    controls.throttle = -0.3
                else:
                    controls.throttle = 0.3
        else:
            if behavior == "Reposition" and car_location.y > 5000:
                controls.boost = False
                if car_velocity.length() > 300:
                    controls.throttle = -0.3
                else:
                    controls.throttle = 0.3

        # Flip into ball
        # TODO: Replace distance with relative locations for the if statement
        if car_location.dist(ball_location) < 300 and ball_grounded and car_grounded and (behavior == "Ball chase" or "Kickoff"):
            return self.begin_smart_flip(packet, car_to_ball)
        
        # Flip for speed
        if (1400 < car_velocity.length() < 1450) and car_grounded and not car_steering and car_location.dist(target_location) > 2750 and (behavior == "Reposition" or "Kickoff"):
        #     # We'll do a front flip if the car is moving at a certain speed.
              return self.begin_front_flip(packet)
        
        # Half flip
        if car_velocity.length() < 1000 and car_grounded and car_to_target.x < -500 and car_location.dist(target_location) > 1500 and abs(car_to_target_angle) < 60:
            return self.begin_half_flip(packet, car_location, car_yaw)
        
        # Boost to gain speed
        if car_velocity.length() != 2300 and car_on_wheels and not car_steering:
            controls.boost = True
        
        # Powerslide to turn quicker
        if abs(controls.steer) >= 1 and car_grounded and car_velocity.length() >= 1000 and car_location.dist(target_location) < 1500:
            controls.throttle = 0.5
            controls.handbrake = True
            # return self.powerslide(packet, controls.steer)
        
        # Land cleanly
        if not car_on_wheels:
            if car_roll < -0.2: controls.roll = 1
            elif car_roll > 0.2: controls.roll = -1
            if car_pitch < -0.2: controls.pitch = 0.5
            elif car_pitch > 0.2: controls.pitch = -0.5
            controls.yaw = yaw_toward_target(my_car, target_location)
            # return self.clean_land(packet, car_roll, car_pitch, car_to_target_angle)
        
        # Slow down if ball is high up
        if not ball_grounded and car_location.dist(ball_location) < 1000 and (behavior == "Ball chase" or "Kickoff"):
            controls.throttle = 0.5
        
        # TESTING: speedflip
        # if car_grounded and car_on_wheels and car_facing_target and not car_steering:
        #     self.begin_speed_flip_smart(packet, car_to_target)
        
        # TESTING: jump for aerial
        # if car_grounded and my_car.boost > 50 and car_facing_target and target_location[2] > 900 and not car_steering:
        #     return self.begin_aerial(packet, car_roll, car_pitch, ball_path)
        
        # TESTING: go for aerial in air (note: this doesn't work very well)
        # if not car_grounded and not car_on_wheels:
        #     return self.continue_aerial(packet, car_roll, car_pitch, ball_path, car_to_target, car_to_target_angle, car_to_target_verticle_angle)
        
        return controls
    

    
    ### INPUTS
    # Flips/Jumps
    # def begin_smart_flip2(self, packet, car_to_ball):
    #     # if car_to_ball[2] > 200 and abs(car_to_ball[1]) < 300 and abs(car_to_ball.x) < 300:
    #     #     return self.begin_double_jump(packet)
    #     if car_to_ball.y > 200:
    #         return self.begin_right_flip(packet)
    #     if car_to_ball.y < -200:
    #         return self.begin_left_flip(packet)
    #     if car_to_ball.y > 100:
    #         return self.begin_diag_right_flip(packet)
    #     if car_to_ball.y < -100:
    #         return self.begin_diag_left_flip(packet)
        
    #     return self.begin_front_flip(packet)
    
    def begin_smart_flip(self, packet, car_to_ball):
        # find angle to ball
        if car_to_ball.x == 0: car_to_ball.x = 0.01
        if car_to_ball.y == 0: car_to_ball.y = 0.01
        yw = car_to_ball.y/(abs(car_to_ball.x) + abs(car_to_ball.y))
        ptch = -car_to_ball.x/(abs(car_to_ball.y) + abs(car_to_ball.x))
        # if abs(yw) < 0.1:
        #     yw = 0
        # if abs(ptch) < 0.1:
        #     ptch = 0
        self.active_sequence = Sequence([
            ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch = ptch, yaw = yw)),
            ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
        
        
        
        
        
    def begin_front_flip(self, packet):
        # Send some quickchat just for fun
        # self.send_quick_chat(team_only=False, quick_chat=QuickChatSelection.Information_IGotIt)
        # Do a front flip. We will be committed to this for a few seconds and the bot will ignore other
        # logic during that time because we are setting the active_sequence.
        self.active_sequence = Sequence([
            ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
    
    def begin_back_flip(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=1)),
            ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
    
    def begin_left_flip(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, yaw=-1)),
            ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
    
    def begin_right_flip(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, yaw=1)),
            ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
    
    def begin_diag_left_flip(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1, yaw=-1)),
            ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
    
    def begin_diag_right_flip(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1, yaw=1)),
            ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
    
    def begin_double_jump(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.30, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
    
    def begin_half_flip(self, packet, car_location, car_yaw):
        if car_location.x > 0:
            x = 1
        else:
            x = -1
        if car_yaw > 0:
            y = 1
        else:
            y = -1
        self.active_sequence = Sequence([
            ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.05, controls=SimpleControllerState(jump=False)),
            ControlStep(duration=0.3, controls=SimpleControllerState(jump=True, pitch=1)),
            ControlStep(duration=0.5, controls=SimpleControllerState(pitch = -1, roll = x * y)),
        ])
        return self.active_sequence.tick(packet)
    
    # Speed Flips
    def begin_speed_flip_smart(self, packet, car_to_target):
        if car_to_target.y > 0:
            return self.begin_speed_flip_right(packet)
        else:
            return self.begin_speed_flip_left(packet)
    
    
    def begin_speed_flip_left(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.02, controls=SimpleControllerState(throttle = 1)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle = 1, jump=True, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=False, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=True, boost = True, pitch=-1, yaw=-1)),
            ControlStep(duration=0.79, controls=SimpleControllerState(throttle = 1, jump=False, boost = True, pitch=1, roll = -1, yaw = -0.5)),
            ControlStep(duration=0.25, controls=SimpleControllerState(throttle = 1, handbrake = True, boost = True)),
        ])
        
    def begin_speed_flip_right(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.02, controls=SimpleControllerState(throttle = 1)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle = 1, jump=True, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=False, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=True, boost = True, pitch=-1, yaw=1)),
            ControlStep(duration=0.79, controls=SimpleControllerState(throttle = 1, jump=False, boost = True, pitch=1, roll = 1, yaw = 0.5)),
            ControlStep(duration=0.25, controls=SimpleControllerState(throttle = 1, handbrake = True, boost = True)),
        ])

        # Return the controls associated with the beginning of the sequence so we can start right away.
        return self.active_sequence.tick(packet)
    
    def begin_aerial(self, packet, car_roll, car_pitch, ball_path):
        self.active_sequence = Sequence([
            ControlStep(duration=0.02, controls=SimpleControllerState()),
            ControlStep(duration=0.01, controls=SimpleControllerState(jump=True)),
            ControlStep(duration=0.14, controls=SimpleControllerState(jump=True, pitch = 1)),
            ControlStep(duration=0.15, controls=SimpleControllerState(jump=False, pitch = 0.75, boost = True)),
            ControlStep(duration=0.20, controls=SimpleControllerState(jump=True, boost = True)),
        ])
        return self.active_sequence.tick(packet)
    
    # def continue_aerial(self, packet, car_roll, car_pitch, ball_path, car_to_target, car_to_target_angle, car_to_target_verticle_angle):
    #     if abs(car_to_target_verticle_angle) <= 10:
    #         ptch = 0
    #     elif car_to_target_verticle_angle > 10:
    #         ptch = -0.2
    #     else:
    #         ptch = 0.2
            
    #     if abs(car_to_target_angle) <= 10:
    #         yw = 0
    #     elif car_to_target_angle > 10:
    #         yw = 0.01
    #     else:
    #         yw = -0.01
    #     self.active_sequence = Sequence([
    #         ControlStep(duration=0.001, controls=SimpleControllerState(yaw=yw, pitch = ptch, boost = True)),
    #     ])
    #     return self.active_sequence.tick(packet)
    
    ### OP KICKOFFS    
    def back_center_kickoff(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.3, controls=SimpleControllerState(throttle = 1, boost = True)),
            ControlStep(duration=0.26, controls=SimpleControllerState(throttle = 1, boost = True, steer = -0.30)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle = 1, jump=True, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=False, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=True, boost = True, pitch=-1, yaw=1)),
            ControlStep(duration=0.79, controls=SimpleControllerState(throttle = 1, jump=False, boost = True, pitch=1, roll = 1, yaw = 0.5)),
            ControlStep(duration=0.25, controls=SimpleControllerState(throttle = 1, handbrake = True, boost = True, steer = 0.25)),
        ])
        return self.active_sequence.tick(packet)

    def back_right_kickoff(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.49, controls=SimpleControllerState(throttle = 1, boost = True, steer = -0.3)),
            ControlStep(duration=0.02, controls=SimpleControllerState(throttle = 1)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle = 1, jump=True, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=False, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=True, boost = True, pitch=-1, yaw=1)),
            ControlStep(duration=0.79, controls=SimpleControllerState(throttle = 1, jump=False, boost = True, pitch=1, roll = 1, yaw = 1)),
            ControlStep(duration=0.20, controls=SimpleControllerState(throttle = 1, handbrake = True, boost = True, steer = 0.05)),
            # ControlStep(duration=0.27, controls=SimpleControllerState(throttle = 1, handbrake = True, boost = True)),
            # ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            # ControlStep(duration=0.01, controls=SimpleControllerState(jump=False)),
            # ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            # ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])
        return self.active_sequence.tick(packet)
    
    def back_left_kickoff(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.49, controls=SimpleControllerState(throttle = 1, boost = True, steer = 0.3)),
            ControlStep(duration=0.02, controls=SimpleControllerState(throttle = 1)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle = 1, jump=True, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=False, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=True, boost = True, pitch=-1, yaw=-1)),
            ControlStep(duration=0.79, controls=SimpleControllerState(throttle = 1, jump=False, boost = True, pitch=1, roll = -1, yaw = -1)),
            ControlStep(duration=0.20, controls=SimpleControllerState(throttle = 1, handbrake = True, boost = True, steer = -0.05)),
            # ControlStep(duration=0.27, controls=SimpleControllerState(throttle = 1, handbrake = True, boost = True)),
            # ControlStep(duration=0.10, controls=SimpleControllerState(jump=True)),
            # ControlStep(duration=0.01, controls=SimpleControllerState(jump=False)),
            # ControlStep(duration=0.2, controls=SimpleControllerState(jump=True, pitch=-1)),
            # ControlStep(duration=0.5, controls=SimpleControllerState()),
        ])
        return self.active_sequence.tick(packet)
    
    def diag_right_kickoff(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.42, controls=SimpleControllerState(throttle = 1, boost = True, steer = 0.35)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle = 1, jump=True, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=False, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=True, boost = True, pitch=-1, yaw=-1)),
            ControlStep(duration=0.79, controls=SimpleControllerState(throttle = 1, jump=False, boost = True, pitch=1, roll = -1, yaw = -0.5)),
            ControlStep(duration=0.10, controls=SimpleControllerState(throttle = 1, handbrake = True, boost = True)),
        ])
        return self.active_sequence.tick(packet)
    
    def diag_left_kickoff(self, packet):
        self.active_sequence = Sequence([
            ControlStep(duration=0.42, controls=SimpleControllerState(throttle = 1, boost = True, steer = -0.35)),
            ControlStep(duration=0.05, controls=SimpleControllerState(throttle = 1, jump=True, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=False, boost = True)),
            ControlStep(duration=0.01, controls=SimpleControllerState(throttle = 1, jump=True, boost = True, pitch=-1, yaw=1)),
            ControlStep(duration=0.79, controls=SimpleControllerState(throttle = 1, jump=False, boost = True, pitch=1, roll = 1, yaw = 0.5)),
            ControlStep(duration=0.10, controls=SimpleControllerState(throttle = 1, handbrake = True, boost = True)),
        ])
        return self.active_sequence.tick(packet)
    
    # TODO: It might be useful to add custom kickoff flip logic based on where opponent is
    def kickoff_flip(self, packet, y, dist, controls, car_to_ball):
        controls.handbrake = True
        controls.boost = True
        if dist <= 500:
            return self.begin_smart_flip(packet, car_to_ball)
    
    
    
    ### BEHAVIORS
    # from blue's then orange's perspective
    # back center: (0.00, -4608.00) (0.00, 4608.00)
    # back right: (-256.00, -3840.00) (256.00, 3840.00)
    # back left: (256.00, -3840.00) (-256.00, 3840.00)
    # diag right: (-2048.00, -2560.00) (2048.00, 2560.00)
    # diag left: (2048.00, -2560.00) (-2048.00, 2560.00)
    def op_kickoffs(self, packet, car_location, coef):
        if round(car_location.x) == 0 and abs(round(car_location.y)) == 4608:
            return self.back_center_kickoff(packet)
        elif round(car_location.x) == (-coef)*256 and round(car_location.y) == (-coef)*3840:
            return self.back_right_kickoff(packet)
        elif round(car_location.x) == (coef)*256 and round(car_location.y) == (-coef)*3840:
            return self.back_left_kickoff(packet)
        elif round(car_location.x) == (-coef)*2048:
            return self.diag_right_kickoff(packet)
        elif round(car_location.x) == (coef)*2048:
            return self.diag_left_kickoff(packet) 
    
    def ball_chase(self, ball_location, ball_path, car_location):
        # distance = car_location.dist(ball_location)
        # if distance > 5000:
        #     distance = 5000
        # if ball_in_future is not None:
        #     target_location = (distance/5000) * Vec3(ball_in_future.physics.location) + (1-distance/5000) * ball_location
        #     self.renderer.draw_line_3d(ball_location, target_location, self.renderer.cyan())
        # else:
        #     target_location = ball_location
        target_location = ball_path
        self.renderer.draw_line_3d(ball_location, target_location, self.renderer.cyan())
        return target_location
            
    def reposition(self, goal_location, car_location):
        target_location = goal_location
        # self.renderer.draw_line_3d(car_location, target_location, self.renderer.cyan())
        return target_location
    
    def leave_net(self, car_location, color, ball_path):
        if abs(ball_path.x) < 800:
            x = ball_path.x
        else:
            x = car_location.x
            if x < 0:
                x + 1
            else:
                x - 1
        if color == 0:
            target_location = Vec3(x, -5119, 0)
        else:
            target_location = Vec3(x, 5119, 0)
        return target_location
    
    def attack(self, ball_location, ball_path, car_location, opp_goal_location):
        # have car push ball towards opponent net (not the corners)
        # if abs(ball_location[0]) < abs(car_location[0]) + 100:
        #     return self.ball_chase(ball_in_future, ball_location, car_location)
        # else:
        if ball_path[0] > 0:
            target_location = Vec3(ball_path[0] + 100, ball_path[1], ball_path[2])
        else:
            target_location = Vec3(ball_path[0] - 100, ball_path[1], ball_path[2])
        return target_location
    
    ### DEBUG
    def debug(self, car_location, target_location, car_debug):
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_string_3d(car_location, 1, 1, car_debug, self.renderer.white())
        self.renderer.draw_rect_3d(target_location, 8, 8, True, self.renderer.cyan(), centered=True)
    
    
### IMPORTANT LOCATIONS
def get_nearest_big_boost(info, packet, car_location, car_orientation):
    nearest_boost_loc = Vec3(0, 0, 999999)
    # loop over all the boosts
    for i, boost in enumerate(info.boost_pads):
        car_to_big = relative_location(car_location, car_orientation, Vec3(boost.location))
        trig = math.atan(car_to_big.y/car_to_big.x)*180/math.pi
        # only want large boosts that haven't been taken
        if boost.is_full_boost and packet.game_boosts[i].is_active:
            # if this boost is closer, save that
            if car_location.dist(Vec3(boost.location)) < car_location.dist(Vec3(nearest_boost_loc)) and 0 < car_to_big.x and abs(trig) < 60:
                nearest_boost_loc = boost.location
    return(nearest_boost_loc)

def get_nearest_small_boost(info, packet, car_location, car_orientation):
    nearest_boost_loc = Vec3(0, 0, 999999)
    # loop over all the boosts
    for i, boost in enumerate(info.boost_pads):
        car_to_small = relative_location(car_location, car_orientation, Vec3(boost.location))
        if car_to_small.x == 0: car_to_small.x = 0.01
        trig = math.atan(car_to_small.y/car_to_small.x)*180/math.pi
        # only want large boosts that haven't been taken
        if not boost.is_full_boost and packet.game_boosts[i].is_active:
            # if this boost is closer, save that
            if car_location.dist(Vec3(boost.location)) < car_location.dist(Vec3(nearest_boost_loc)) and 0 < car_to_small.x and abs(trig) < 30:
                nearest_boost_loc = boost.location
    return(nearest_boost_loc)

# TODO: Fix false positives with angle
# It would probably be easier to figure out the x coordinates the ball needs to be within given its y
def triangle(goal_location, left_post, right_post, ball_path, ball_velocity, car_location):
    triangle_length = goal_location.y - ball_path.y
    triangle_width = goal_location.x - ball_path.x
    triangle_hyp = (triangle_length**2 + triangle_width**2)**(1/2)
    ball_path_angle_to_net = math.atan2(triangle_length, triangle_width)
    new_hyp = triangle_hyp*(1 + ball_velocity.length()/15000)
    new_y = new_hyp * math.sin(ball_path_angle_to_net)
    new_x = new_hyp * math.cos(ball_path_angle_to_net)
    
    triangle_car_length = goal_location.y - car_location.y
    triangle_width_left = left_post.x - car_location.x
    car_angle_to_left = math.atan2(triangle_car_length, triangle_width_left)
    triangle_width_right = right_post.x - car_location.x
    car_angle_to_right = math.atan2(triangle_car_length, triangle_width_right)
    
    ball_triangle_width_left = left_post.x - ball_path.x
    ball_angle_to_left = math.atan2(triangle_length, ball_triangle_width_left)
    ball_triangle_width_right = right_post.x - ball_path.x
    ball_angle_to_right = math.atan2(triangle_length, ball_triangle_width_right)
    
    car_to_ball_length = ball_path.y - car_location.y
    car_to_ball_width = ball_path.x - car_location.x
    car_to_ball_angle = math.atan2(car_to_ball_length, car_to_ball_width)
    
    get_positioning_on_ball = Vec3(goal_location.x - new_x, goal_location.y - new_y, 0)
    if get_positioning_on_ball.x > 5100:
        get_positioning_on_ball.x = 5100
    elif get_positioning_on_ball.x < -5100:
        get_positioning_on_ball.x = -5100
    
    
    angle = False
    if ball_path.dist(goal_location) < car_location.dist(goal_location) and (car_angle_to_left < car_to_ball_angle < car_angle_to_right or car_angle_to_right < car_to_ball_angle < car_angle_to_left):
        angle = True
        
    return (car_angle_to_left, car_angle_to_right, get_positioning_on_ball, angle)