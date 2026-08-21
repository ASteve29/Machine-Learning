#pragma once

#include <vector>
#include <random>
#include <memory>
#include <raylib.h>
#include "physics_objects.h"

struct rect {
    float x, y, width, height;

    bool contains(Vector2 point) const {
        return (point.x >= x && 
                point.x <= x + width && 
                point.y >= y && 
                point.y <= y + height);
    }
};

class QuadtreeNode {
    rect boundary;

    float mass;
    Vector2 centre_of_mass;

    Body* body;
    bool is_divided;

    std::unique_ptr<QuadtreeNode> nw;
    std::unique_ptr<QuadtreeNode> ne;
    std::unique_ptr<QuadtreeNode> sw;
    std::unique_ptr<QuadtreeNode> se;

    QuadtreeNode(rect bounds);

    bool insert(Body* new_body);
    void subdivide();
    void compute_mass_distribution();

    Vector2 calculate_gravity(const Body* target, float theta, float G) const;
};
