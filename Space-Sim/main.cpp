#include <raylib.h>
#include <vector>
#include <cmath>
#include <random>
#include <raymath.h>

#include "physics_objects.h"
#include "quadtree.h"
#include "matrix.h"
#include "neural_network.h"

const float G = 1500.0f;
const float SOFTENING = 500.0f;

float compute_step_reward(
    float prev_dist, 
    float current_dist, 
    const std::vector<bool>& actions, 
    const Ship& ship, 
    Vector2 target_pos,
    float dt
) {
    float reward = 0.0f;

    // 1. Distance Delta (Progress toward target)
    // Moving closer gives positive reward; drifting away gives negative penalty
    float delta_dist = prev_dist - current_dist;
    reward += delta_dist * 0.1f; 

    reward -= dt;

    // 3. Heading Alignment Reward (Bonus for facing toward the target)
    Vector2 dir_to_target = Vector2Normalize(Vector2Subtract(target_pos, ship.position));
    Vector2 facing_dir = { cosf(ship.rotation), sinf(ship.rotation) };
    float alignment = Vector2DotProduct(facing_dir, dir_to_target); // Ranges from -1.0 to +1.0
    
    if (alignment > 0.0f) {
        reward += alignment * 0.5f; // Small bonus for keeping nose pointed at target
    }

    return reward;
}

int main() {
    const int screenWidth = 1800;
    const int screenHeight = 900;
    InitWindow(screenWidth, screenHeight, "2D N-Body Orbit Simulation");
    SetTargetFPS(0);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> distX(0.0f, (float)screenWidth);
    std::uniform_real_distribution<float> distY(0.0f, (float)screenHeight);
    std::uniform_real_distribution<float> mass(5.0f, 50.0f);
    std::uniform_real_distribution<float> radius(2.5f, 5.0f);
    std::uniform_real_distribution<float> vel(-30.0f, 30.0f);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    const size_t INPUT_SIZE  = 38;
    const size_t HIDDEN_SIZE = 64; 
    const size_t OUTPUT_SIZE = 4;
    const float WEIGHT_INIT  = 0.1f;

    PolicyGradient brain(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, WEIGHT_INIT);

    Ship ship(500.0f, 400.0f);
    std::vector<Body> bodies;

    std::uniform_real_distribution<float> targetDistX(100.0f, (float)screenWidth - 100.0f);
    std::uniform_real_distribution<float> targetDistY(100.0f, (float)screenHeight - 100.0f);

    Vector2 target_pos = { targetDistX(rng), targetDistY(rng) };
    float episode_timer = 0.0f;
    int simulation_speed = 1;

    auto reset_simulation = [&]() {
        ship.position = { 500.0f, 400.0f };
        ship.velocity = { 0.0f, 0.0f };
        ship.rotation = 0.0f;
        
        bodies.clear();
        for (int i = 0; i < 20; i++) {
            bodies.push_back({
                {distX(rng), distY(rng)},
                {vel(rng), vel(rng)},
                {0.0f, 0.0f},
                mass(rng),
                radius(rng),
                BLUE
            });
        }

        target_pos = { targetDistX(rng), targetDistY(rng) };
        episode_timer = 0.0f;
    };

    // Initial population of bodies
    reset_simulation();

    int sim_num = 0;

    while (!WindowShouldClose()) {
        if (sim_num%100 == 0) {
            simulation_speed = 1;
        } else {
            simulation_speed = 1000;
        }

        float fixed_dt = 0.01667f; 

        for (int step = 0; step < simulation_speed; ++step) {
            episode_timer += fixed_dt;

            float prev_dist_to_target = Vector2Distance(ship.position, target_pos);

            for (const auto& planet : bodies) {
                Vector2 dir = Vector2Subtract(planet.position, ship.position);
                float distSq = Vector2LengthSqr(dir) + SOFTENING;
                float force_mag = (G * ship.mass * planet.mass) / distSq;
                Vector2 norm_dir = Vector2Normalize(dir);
                ship.apply_force({ norm_dir.x * force_mag, norm_dir.y * force_mag });
            }

            for (size_t i = 0; i < bodies.size(); i++) {
                for (size_t j = i + 1; j < bodies.size(); j++) {
                    Vector2 dir = Vector2Subtract(bodies[j].position, bodies[i].position);
                    float distSq = Vector2LengthSqr(dir) + SOFTENING;
                    float force_mag = (G * bodies[i].mass * bodies[j].mass) / distSq;
                    Vector2 norm_dir = Vector2Normalize(dir);
                    bodies[i].apply_force({ norm_dir.x * force_mag, norm_dir.y * force_mag });
                    bodies[j].apply_force({ -norm_dir.x * force_mag, -norm_dir.y * force_mag });
                }
            }
            std::vector<float> inputs = ship.get_sensor_input(bodies, target_pos, (float)screenWidth, (float)screenHeight);
            std::vector<float> action_probs = brain.forward(inputs);

            std::vector<bool> actions(4, false);
            for (size_t i = 0; i < 4; ++i) {
                actions[i] = (prob_dist(rng) < action_probs[i]);
            }
            ship.fire_thrusters(actions);

            ship.update(fixed_dt);
            for (auto& b : bodies) {
                b.update(fixed_dt);
            }

            float current_dist_to_target = Vector2Distance(ship.position, target_pos);

            float step_reward = compute_step_reward(
                                prev_dist_to_target, 
                                current_dist_to_target, 
                                actions, 
                                ship, 
                                target_pos,
                                fixed_dt
                            );

            bool crashed_into_planet = false;
            for (const auto& planet : bodies) {
                if (CheckCollisionCircles(ship.position, ship.radius, planet.position, planet.radius)) {
                    crashed_into_planet = true;
                    break;
                }
            }

            bool target_reached = (current_dist_to_target < 25.0f);
            bool out_of_bounds = (ship.position.x < 0 || ship.position.x > screenWidth || 
                                  ship.position.y < 0 || ship.position.y > screenHeight);
            bool timed_out = (episode_timer > 30.0f);

            brain.train(actions, step_reward, 0.01f);

            if (target_reached || out_of_bounds || crashed_into_planet || timed_out) {
                float terminal_reward = 0.0f;
                if (target_reached)       terminal_reward = +2.0f;
                else if (crashed_into_planet) terminal_reward = -1.5f;
                else if (out_of_bounds)   terminal_reward = -1.0f;
                else if (timed_out)       terminal_reward = -0.5f;

                brain.train(actions, terminal_reward, 0.01f);

                sim_num++;

                reset_simulation();
            }
        }

        // Render Frame
        BeginDrawing();
        ClearBackground(BLACK);

        DrawCircleV(target_pos, 8.0f, GREEN);
        DrawCircleLines((int)target_pos.x, (int)target_pos.y, 15.0f, GREEN);

        for (const auto& b : bodies) {
            DrawCircleV(b.position, b.radius, b.colour);
        }

        ship.draw();

        DrawText(TextFormat("Speed: %dx (Hold SPACE for 50x)", simulation_speed), 10, 40, 20, RAYWHITE);
        DrawText(TextFormat("Generation number: %d", sim_num), 10, 70, 20, RAYWHITE);
        DrawFPS(10, 10);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
