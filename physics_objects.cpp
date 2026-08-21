#include "physics_objects.h"
#include <raymath.h>
#include <algorithm>

void Body::apply_force(Vector2 force) {
    acceleration.x += force.x / mass;
    acceleration.y += force.y / mass;
}

void Body::update(float dt) {
    velocity.x += 0.5f * acceleration.x * dt;
    velocity.y += 0.5f * acceleration.y * dt;

    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    acceleration = {0.0f, 0.0f};
}

void Ship::fire_thrusters(const std::vector<bool>& actions) {
    if (actions.size() < 4) return;
    float turn_speed = 3.0f; // Radians per sec
    float dt = GetFrameTime();

    if (actions[2]) rotation -= turn_speed * dt;
    if (actions[3]) rotation += turn_speed * dt;

    Vector2 forward_dir = { cosf(rotation), sinf(rotation) };

    if (actions[0]) {
        apply_force({ forward_dir.x * thrust_power, forward_dir.y * thrust_power });
    }
    if (actions[1]) {
        apply_force({ -forward_dir.x * (thrust_power * 0.5f), -forward_dir.y * (thrust_power * 0.5f) });
    }
}

std::vector<float> Ship::get_sensor_input(
    const std::vector<Body>& bodies, 
    Vector2 target_pos,
    float world_width,
    float world_height,
    float max_speed,
    float max_turn_rate,
    float max_mass
) const {
    std::vector<float> inputs;
    inputs.reserve(38);

    inputs.push_back(velocity.x/max_speed);
    inputs.push_back(velocity.y/max_speed);
    inputs.push_back(cosf(rotation));
    inputs.push_back(sinf(rotation));
    inputs.push_back(angular_vel/max_turn_rate);

    inputs.push_back(target_pos.x - position.x/world_width);
    inputs.push_back(target_pos.y - position.y/world_height);

    float max_diag = sqrtf(world_width * world_width + world_height * world_height);
    inputs.push_back(Vector2Distance(position, target_pos) / max_diag);

    auto push_body_features = [&](const Body* b) {
        if (b != nullptr) {
            inputs.push_back((b->position.x - position.x) / world_width);
            inputs.push_back((b->position.y - position.y) / world_height);
            inputs.push_back((b->velocity.x - velocity.x) / max_speed);
            inputs.push_back((b->velocity.y - velocity.y) / max_speed);
            inputs.push_back(b->mass / max_mass);
        } else {
            inputs.push_back(2.0f);
            inputs.push_back(2.0f);
            inputs.push_back(0.0f);
            inputs.push_back(0.0f);
            inputs.push_back(0.0f);
        }
    };

    std::vector<const Body*> gravity_sorted;
    for(const auto& b : bodies) gravity_sorted.push_back(&b);

    std::sort(gravity_sorted.begin(), gravity_sorted.end(), [this](const Body* a, const Body* b) {
        float distSq_a = Vector2DistanceSqr(position, a->position) + 1e-5f;
        float distSq_b = Vector2DistanceSqr(position, b->position) + 1e-5f;
        return (a->mass / distSq_a) > (b->mass / distSq_b);
    });

    for(size_t i = 0; i < 3; ++i) {
        push_body_features(i < gravity_sorted.size() ? gravity_sorted[i] : nullptr);
    }

    std::vector<const Body*> proximity_sorted;
    for (const auto& b : bodies) proximity_sorted.push_back(&b);

    std::sort(proximity_sorted.begin(), proximity_sorted.end(), [this](const Body* a, const Body* b) {
        return Vector2DistanceSqr(position, a->position) < Vector2DistanceSqr(position, b->position);
    });

    for (size_t i = 0; i < 3; ++i) {
        push_body_features(i < proximity_sorted.size() ? proximity_sorted[i] : nullptr);
    }

    return inputs;
}

void Ship::draw() const{
    float length = radius * 1.25f;
    float width = radius * 0.5f;

    Vector2 nose = {
        position.x + cosf(rotation) * length,
        position.y + sinf(rotation) * length
    };

    Vector2 right = { -sinf(rotation), cosf(rotation) };

    Vector2 rear_left = {
        position.x - cosf(rotation) * (length * 0.5f) - right.x * width,
        position.y - sinf(rotation) * (length * 0.5f) - right.y * width
    };

    Vector2 rear_right = {
        position.x - cosf(rotation) * (length * 0.5f) + right.x * width,
        position.y - sinf(rotation) * (length * 0.5f) + right.y * width
    };

    DrawTriangle(nose, rear_left, rear_right, colour);
    DrawTriangleLines(nose, rear_left, rear_right, DARKGRAY);
}