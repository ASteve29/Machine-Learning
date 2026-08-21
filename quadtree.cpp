#include "quadtree.h"
#include <cmath>

// Constructor
QuadtreeNode::QuadtreeNode(rect bounds) 
    : boundary(bounds), mass(0.0f), centre_of_mass({0.0f, 0.0f}), 
      body(nullptr), is_divided(false) {}

// Subdivide the node into 4 equal child quadrants
void QuadtreeNode::subdivide() {
    float half_w = boundary.width / 2.0f;
    float half_h = boundary.height / 2.0f;
    float x = boundary.x;
    float y = boundary.y;

    // TODO: Construct nw, ne, sw, se using std::make_unique<QuadtreeNode>(...)
    // Example for top-left (NW):
    // nw = std::make_unique<QuadtreeNode>(rect{ x, y, half_w, half_h });
    
    // Set division flag
    is_divided = true;
}

// Insert a body into the quadtree
bool QuadtreeNode::insert(Body* new_body) {
    // 1. Return false if the body is outside this boundary

    // 2. If this node has no body and is not divided, store it here and return true

    // 3. If this node is not divided yet, subdivide and push the existing body down

    // 4. Try inserting new_body into nw, ne, sw, or se

    return false;
}

// Recursively compute total mass and centre of mass (Barnes-Hut)
void QuadtreeNode::compute_mass_distribution() {
    if (!is_divided) {
        // Base case: If leaf node with a body, set mass and centre of mass from body
        if (body != nullptr) {
            mass = body->mass;
            centre_of_mass = body->position;
        }
        return;
    }

    // Recursive case: Branch node
    // 1. Call compute_mass_distribution() on all 4 children
    // 2. Combine mass from children
    // 3. Calculate weighted centre_of_mass: sum(child_mass * child_centre) / mass
}

// Calculate gravitational force on a target body using Barnes-Hut criterion
Vector2 QuadtreeNode::calculate_gravity(const Body* target, float theta, float G) const {
    Vector2 net_force = { 0.0f, 0.0f };

    // 1. If empty node, return zero force

    // 2. If this is a leaf node containing a body (and it's not target itself):
    //    Calculate standard Newton's gravity force vector

    // 3. Calculate distance 'd' from target to this node's centre_of_mass
    //    Calculate width 's' of this node's boundary

    // 4. Check Barnes-Hut criterion: (s / d) < theta
    //    IF TRUE: Treat node as a single point mass! Calculate gravity using mass and centre_of_mass
    //    IF FALSE: Recurse down into all 4 children and accumulate their forces

    return net_force;
}