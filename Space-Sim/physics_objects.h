
#pragma once

#include <raylib.h>
#include <vector>
#include <cmath>
#include <random>

struct Body {
    Vector2 position;
    Vector2 velocity;
    Vector2 acceleration;
    float mass;
    float radius;
    Color colour;

    void apply_force(Vector2 force);
    void update(float dt);
};

class Ship : public Body {
    public:
        float rotation;
        float angular_vel;
        float thrust_power;

        Ship(float start_x, float start_y, float start_mass = 10.0f, float start_radius = 10.0f) : rotation(0.0f), angular_vel(0.0f), thrust_power(2000.0f){
            position = {start_x, start_y};
            velocity = {0.0f, 0.0f};
            acceleration = {0.0f, 0.0f};
            mass = start_mass;
            radius = start_radius;
        }

        void fire_thrusters(const std::vector<bool>& actions);
        std::vector<float> get_sensor_input(
            const std::vector<Body>& bodies, 
            Vector2 target_pos,
            float world_width = 1000.0f,
            float world_height = 800.0f,   
            float max_speed = 500.0f,
            float max_turn_rate = 10.0f,
            float max_mass = 100.0f
        ) const;

        void draw() const;
};
