//
// Created by Andy Wang
//

#ifndef TRAJECTORY_HPP
#define TRAJECTORY_HPP

#include <cmath>
#include <cassert>
#include <array>
#include <bitset>
#include <fstream>
#include <random>

#include <box2d/box2d.h>
#include <robox2d/simu.hpp>
#include <robox2d/robot.hpp>
#include <robox2d/common.hpp>
#include <robox2d/gui/magnum/graphics.hpp>

// pseudo random number generator
namespace rng {
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> rng(0, 1);
}

class Arm : public robox2d::Robot {
public:
  
    Arm(std::shared_ptr<b2World> world, int &ball_moved_by_noise, bool apply_force = false)
    {
        size_t nb_joints = 4;
        float arm_length = 1.5f;
        float seg_length = arm_length / (float) nb_joints;

        // create walls one by one as GUI needs seperate bodies, so cannot use one body for multiple fixtures
        // specify width and height to each side from a central location
        if (Params::sim::enable_graphics)
        {
            b2Body* ceiling = robox2d::common::createBox(world, {Params::sim::ROOM_W / 2, 0.01}, b2_staticBody, {Params::sim::ROOM_W / 2, Params::sim::ROOM_H, 0.f});
            b2Body* floor = robox2d::common::createBox(world, {Params::sim::ROOM_W / 2, 0.01}, b2_staticBody, {Params::sim::ROOM_W / 2, 0.f, 0.f});
            b2Body* right = robox2d::common::createBox(world, {0.01, Params::sim::ROOM_H / 2}, b2_staticBody, {Params::sim::ROOM_W, Params::sim::ROOM_H / 2, 0.f});
            b2Body* left = robox2d::common::createBox(world, {0.01, Params::sim::ROOM_H / 2}, b2_staticBody, {0, Params::sim::ROOM_H / 2, 0.f});
        }
        else // if not using the GUI, create one body with 4 walls for faster sim
            {b2Body* room = robox2d::common::createRoom(world, {Params::sim::ROOM_W, Params::sim::ROOM_H});}

        // base in the center of the room
        b2Body* body = robox2d::common::createBox(world, {arm_length*0.025f, arm_length*0.025f}, b2_staticBody, {Params::sim::ROOM_W / 2, Params::sim::ROOM_H / 2,0.f});
        b2Vec2 anchor = body->GetWorldCenter();
        
        for(size_t i{0}; i < nb_joints; ++i)
        {
            float density = 1.0f/std::pow(1.5,i);
            _end_effector = robox2d::common::createBox(world, {seg_length*0.5f, arm_length*0.01f}, b2_dynamicBody, {(0.5f+i)*seg_length + Params::sim::ROOM_W / 2, Params::sim::ROOM_H / 2,0.0f}, density);
            this->_servos.push_back(std::make_shared<robox2d::common::Servo>(world,body, _end_effector, anchor));

            body = _end_effector;
            anchor = _end_effector->GetWorldCenter() + b2Vec2(seg_length*0.5, 0.0f);
        }

        b2Body* balltomove1 = robox2d::common::createShovableCircle(world, Params::sim::radius, b2_dynamicBody, {Params::sim::ball1_x, Params::sim::ball1_y, 0.f}, 0.8f);
        b2Body* balltomove2 = robox2d::common::createShovableCircle(world, Params::sim::radius, b2_dynamicBody, {Params::sim::ball2_x, Params::sim::ball2_y, 0.f}, 0.8f);
        
        std::vector<b2Body*> objects{balltomove1, balltomove2};
        // random movements
        if (apply_force)
        {
            // use rng to generate force
            float force_x = (rng::rng(rng::gen) - 0.5) * Params::sim::max_force;
            float force_y = (rng::rng(rng::gen) - 0.5) * Params::sim::max_force;
            b2Vec2 force{force_x, force_y};

            ball_moved_by_noise = (rng::rng(rng::gen) > 0.5) ? 0 : 1;
            objects[ball_moved_by_noise]->ApplyForce(force, objects[ball_moved_by_noise]->GetWorldCenter(), true);
        }
    }
  
b2Vec2 get_end_effector_pos(){return _end_effector->GetWorldCenter();}
  
private:
b2Body* _end_effector;
};

FIT_QD(Trajectory)
{
    public:
    Trajectory(): _params(Params::qd::gen_dim), _full_trajectory(Params::sim::full_trajectory_length), 
                  _image(Params::nov::discretisation * Params::nov::discretisation),
                  _undisturbed_image(Params::nov::discretisation * Params::nov::discretisation) {
        for (Eigen::VectorXf &traj : _trajectories)
            {traj.resize(Params::sim::num_trajectory_elements);}
    
        for (Eigen::VectorXf &traj : _undisturbed_trajectories)
            {traj.resize(Params::sim::num_trajectory_elements);}
    }

    template <typename Indiv> 
    void eval(Indiv & ind){

        for (size_t i = 0; i < ind.size(); ++i)
            _params[i] = ind.data(i);

        // noise in environment
        float prob = rng::rng(rng::gen);
        _apply_force = prob < Params::random::pct_random;

        #ifdef AURORA
        simulate(_params, _ball_moved_by_noise, _apply_force);
        generate_image();
        // Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
        // std::cout << _image.format(CommaInitFmt)  << "\n" << std::endl;

        #endif
               
        // FITNESS: constant because we're interested in exploration
        this->_value = -1;
    }
    
    // generate trajectories during the algorithm
    void simulate(Eigen::VectorXd &ctrl_pos, int &ball_moved_by_noise, bool &apply_force){
        robox2d::Simu simu;
        simu.add_floor();
        
        auto rob = std::make_shared<Arm>(simu.world(), ball_moved_by_noise, apply_force);
        Eigen::VectorXd controller = ctrl_pos.segment(0, 4);
        auto ctrl = std::make_shared<robox2d::control::ConstantPos>(controller);
        rob->add_controller(ctrl);

        controller = ctrl_pos.segment(4, 4);
        auto ctrl2 = std::make_shared<robox2d::control::ConstantPos>(controller);
        rob->add_controller(ctrl2);
        simu.add_robot(rob);

        if (Params::sim::enable_graphics)
        {
            auto graphics = std::make_shared<robox2d::gui::Graphics<>>(simu.world());
            simu.set_graphics(graphics);
        }
        simu.run(Params::sim::sim_duration, _trajectories, _full_trajectory, Params::sim::trajectory_length);


        // Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
        // std::cout << _image.format(CommaInitFmt)  << "\n" << std::endl;
    }

    // generate full trajectory for diversity calc and loss tracking
    void simulate(Eigen::VectorXd &ctrl_pos){
    // simulating
        robox2d::Simu simu;
        simu.add_floor();

        // no noise
        auto rob = std::make_shared<Arm>(simu.world(), _ball_moved_by_noise);
        Eigen::VectorXd controller = ctrl_pos.segment(0, 4);
        auto ctrl = std::make_shared<robox2d::control::ConstantPos>(controller);
        rob->add_controller(ctrl);

        controller = ctrl_pos.segment(4, 4);
        auto ctrl2 = std::make_shared<robox2d::control::ConstantPos>(controller);
        rob->add_controller(ctrl2);
        simu.add_robot(rob);

        // if (Params::sim::enable_graphics)
        // {
        //     auto graphics = std::make_shared<robox2d::gui::Graphics<>>(simu.world());
        //     simu.set_graphics(graphics);
        // }

        simu.run(Params::sim::sim_duration, _undisturbed_trajectories, _full_trajectory, Params::sim::trajectory_length);
    }

    void create_observations()
    {
        if (!_observations_generated)
        {
            // create trajectories
            simulate(_params, _ball_moved_by_noise, _apply_force);

            generate_image();

            // for diversity and loss tracking generate only the real trajectory without any randomness 
            if (_apply_force)
            {
                simulate(_params);
                generate_undisturbed_image();
            }
            else // if no noise in simulation already
            {
                _undisturbed_trajectories = _trajectories;
                _undisturbed_image = _image;
            }
            _observations_generated = true;
        }
    }

    void create_undisturbed_observations()
    {
        if (!_undisturbed_trajectories_generated)
        {
            // for diversity and loss tracking generate only the real trajectory without any randomness 
            if (_apply_force)
            {
                simulate(_params);
                generate_undisturbed_image();
            }
            else // if no noise in simulation already
            {
                _undisturbed_trajectories = _trajectories;
                _undisturbed_image = _image;
            }
            _undisturbed_trajectories_generated = true;
        }
    }
    
    void generate_image()
        {generate_image_last_position();}

    void generate_undisturbed_image()
        {generate_undisturbed_image_last_position();}

    void find_ball_pixel_indices(double x, double y, std::vector<int> &indices)
    {
        int index_x = x / Params::nov::discrete_length_x;
        int index_y = y / Params::nov::discrete_length_y;

        // find the ball location
        int bucket = index_x + index_y * Params::nov::discretisation;
        
        // placing center to the right of the bucket
        // 4x1 central row
        if (x - static_cast<float>(index_x * Params::nov::discrete_length_x) > (Params::nov::discrete_length_x / 2))
        {
            indices[0] = (index_x - 1 >= 0) ? bucket - 1 : -1;
            indices[1] = bucket;
            indices[2] = (index_x + 1 <= Params::nov::discretisation - 1) ? bucket + 1 : -1;
            indices[3] = (index_x + 2 <= Params::nov::discretisation - 1) ? bucket + 2 : -1;
        }
        else // place to the left
        {
            indices[0] = (index_x - 2 >= 0) ? bucket - 2 : -1;
            indices[1] = (index_x - 1 >= 0) ? bucket - 1 : -1;
            indices[2] = bucket;
            indices[3] = (index_x + 1 <= Params::nov::discretisation - 1) ? bucket + 1 : -1;
        }

        // place second central row above, because ball is closer to the top
        if (y - static_cast<float>(index_y * Params::nov::discrete_length_y) > (Params::nov::discrete_length_y / 2))
        {
            // central row 4x1
            if (index_y + 1 <= Params::nov::discretisation - 1)
            {
                indices[4] = (indices[0] != -1) ? indices[0] + Params::nov::discretisation : -1;
                indices[5] = (indices[1] != -1) ? indices[1] + Params::nov::discretisation : -1;
                indices[6] = (indices[2] != -1) ? indices[2] + Params::nov::discretisation : -1;
                indices[7] = (indices[3] != -1) ? indices[3] + Params::nov::discretisation : -1;

                // top row 2x1
                if (index_y + 2 <= Params::nov::discretisation - 1)
                {
                    indices[8] = (indices[5] != -1) ? indices[5] + Params::nov::discretisation : -1;
                    indices[9] = (indices[6] != -1) ? indices[6] + Params::nov::discretisation : -1;
                }
            }
            // bottom row 2x1
            if (index_y - 1 >= 0)
            {
                indices[10] = (indices[1] != -1) ? indices[1] - Params::nov::discretisation : -1;
                indices[11] = (indices[2] != -1) ? indices[2] - Params::nov::discretisation : -1;
            }
        }
        else // going down
        {
            // central row 4x1
            if (index_y - 1 >= 0)
            {
                indices[4] = (indices[0] != -1) ? indices[0] - Params::nov::discretisation : -1;
                indices[5] = (indices[1] != -1) ? indices[1] - Params::nov::discretisation : -1;
                indices[6] = (indices[2] != -1) ? indices[2] - Params::nov::discretisation : -1;
                indices[7] = (indices[3] != -1) ? indices[3] - Params::nov::discretisation : -1;

                // bottom row 2x1
                if (index_y - 2 >= 0)
                {
                    indices[8] = (indices[5] != -1) ? indices[5] - Params::nov::discretisation : -1;
                    indices[9] = (indices[6] != -1) ? indices[6] - Params::nov::discretisation : -1;
                }
            }
            // top row 2x1
            if (index_y + 1 <= Params::nov::discretisation - 1)
            {
                indices[10] = (indices[1] != -1) ? indices[1] + Params::nov::discretisation : -1;
                indices[11] = (indices[2] != -1) ? indices[2] + Params::nov::discretisation : -1;
            }
        }

    }

    void generate_image_last_position()
    {
        // initialise image
        _image.fill(0);
        for (int j{0}; j < 2; ++j)
        {
            double x = _trajectories[j][Params::sim::num_trajectory_elements - 2];
            double y = _trajectories[j][Params::sim::num_trajectory_elements - 1];

            std::vector<int> indices(12, -1);

            find_ball_pixel_indices(x, y, indices);
            
            for (int &index : indices)
            {
                if (index != -1)
                    {_image[index] = 1;}
            }
        }

    }

    void generate_image_long_exposure()
    {
        // initialise image
        _image.fill(0);
        for (int i {0}; i < Params::sim::num_trajectory_elements; i += 2)
        {
            for (int j{0}; j < 2; ++j)
            {
                double x = _trajectories[j](i);
                double y = _trajectories[j](i + 1);

                std::vector<int> indices(12, -1);

                find_ball_pixel_indices(x, y, indices);
                
                for (int &index : indices)
                {
                    if (index != -1)
                        {_image[index] = 1;}
                }
            }
        }
    }

    void generate_undisturbed_image_last_position()
    {
        // initialise image
        _undisturbed_image.fill(0);
        for (int j{0}; j < 2; ++j)
        {
            double x = _undisturbed_trajectories[j][Params::sim::num_trajectory_elements - 2];
            double y = _undisturbed_trajectories[j][Params::sim::num_trajectory_elements - 1];

            std::vector<int> indices(12, -1);

            find_ball_pixel_indices(x, y, indices);

            for (int &index : indices)
            {
                if (index != -1)
                    {_undisturbed_image[index] = 1;}
            }
        }
        // std::cout << "HELLO";
        // Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "");
        // std::cout << _undisturbed_trajectories[0].format(CommaInitFmt) << std::endl;
        // std::cout << "\n";
        // std::cout << _undisturbed_image.format(CommaInitFmt);
        // exit(0);
    }

    void generate_undisturbed_image_longexposure()
    {
        // initialise image
        _undisturbed_image.fill(0);
        for (int i {0}; i < Params::sim::num_trajectory_elements; i += 2)
        {
            for (int j{0}; j < 2; ++j)
            {
                double x = _undisturbed_trajectories[j](i);
                double y = _undisturbed_trajectories[j](i + 1);

                std::vector<int> indices(12, -1);

                find_ball_pixel_indices(x, y, indices);

                for (int &index : indices)
                {
                    if (index != -1)
                        {_undisturbed_image[index] = 1;}
                }
            }
        }
    }

    void calculate_distance(float &distance1, float &distance2, float &undist_distance1, float &undist_distance2)
    {
        Eigen::VectorXf manhattan_dist = _trajectories[0].segment(0, Params::sim::num_trajectory_elements - 2) - 
                                         _trajectories[0].segment(2, Params::sim::num_trajectory_elements - 2);

        distance1 = 0;
        for (int i{0}; i < manhattan_dist.size(); i+=2)
            {distance1 += manhattan_dist.segment<2>(i).norm();}

        manhattan_dist = _trajectories[1].segment(0, Params::sim::num_trajectory_elements - 2) - 
                         _trajectories[1].segment(2, Params::sim::num_trajectory_elements - 2);

        distance2 = 0;
        for (int i{0}; i < manhattan_dist.size(); i+=2)
            {distance2 += manhattan_dist.segment<2>(i).norm();}

        manhattan_dist = _undisturbed_trajectories[0].segment(0, Params::sim::num_trajectory_elements - 2) - 
                         _undisturbed_trajectories[0].segment(2, Params::sim::num_trajectory_elements - 2);

        undist_distance1 = 0;
        for (int i{0}; i < manhattan_dist.size(); i+=2)
            {undist_distance1 += manhattan_dist.segment<2>(i).norm();}

        manhattan_dist = _undisturbed_trajectories[1].segment(0, Params::sim::num_trajectory_elements - 2) - 
                         _undisturbed_trajectories[1].segment(2, Params::sim::num_trajectory_elements - 2);

        undist_distance2 = 0;
        for (int i{0}; i < manhattan_dist.size(); i+=2)
            {undist_distance2 += manhattan_dist.segment<2>(i).norm();}

    }

    void get_end_positions(float &ball_1_x, float &ball_1_y, float &ball_2_x, float &ball_2_y,
                           float &undist_ball_1_x, float &undist_ball_1_y, float &undist_ball_2_x, float &undist_ball_2_y)
    {
        ball_1_x = _trajectories[0][Params::sim::num_trajectory_elements - 2];
        ball_1_y = _trajectories[0][Params::sim::num_trajectory_elements - 1];
        ball_2_x = _trajectories[1][Params::sim::num_trajectory_elements - 2];
        ball_2_y = _trajectories[1][Params::sim::num_trajectory_elements - 1];
        undist_ball_1_x = _undisturbed_trajectories[0][Params::sim::num_trajectory_elements - 2];
        undist_ball_1_y = _undisturbed_trajectories[0][Params::sim::num_trajectory_elements - 1];
        undist_ball_2_x = _undisturbed_trajectories[1][Params::sim::num_trajectory_elements - 2];
        undist_ball_2_y = _undisturbed_trajectories[1][Params::sim::num_trajectory_elements - 1];
    }

    template<typename block_t>
    void get_flat_observations(block_t &data) const 
    {
        for (size_t row {0}; row < (Params::random::max_num_random + 1); ++row)
        {   
            for (size_t i{0}; i < Params::sim::num_trajectory_elements; ++i)
                {data(row, i) = _trajectories[row](i);}
        }
    }

    int get_bucket_index(double discrete_length_x, double discrete_length_y, int discretisation) const
    {
        int bucket_x = _full_trajectory[Params::sim::full_trajectory_length - 2] / discrete_length_x;
        int bucket_y = _full_trajectory[Params::sim::full_trajectory_length - 1] / discrete_length_y;
        return bucket_y * discretisation + bucket_x;
    }

    Eigen::VectorXf get_undisturbed_trajectory() const
    {return _undisturbed_trajectories[0];}

    Eigen::VectorXf &get_image()
    {return _image;}

    Eigen::VectorXf &get_undisturbed_image()
    {return _undisturbed_image;}

    float &entropy() 
    {return _m_entropy;}

    int get_idx_ball_moved_by_noise() const
    {return _ball_moved_by_noise;}

    Eigen::VectorXd &params()
    {return _params;}

    bool is_influenced_by_noise() const
    {return _apply_force;}

    // generates images from the trajectories fed into the function
    // void generate_image_sequence()
    // {
    //     for (int i {0}; i < Params::sim::num_trajectory_elements; i += 2)
    //     {
    //         // initialise image
    //         int image_num {i / 2};
    //         _image_frames[image_num] = Eigen::Matrix<float, Params::nov::discretisation, Params::nov::discretisation>::Zero();
    //         for (auto &traj : _trajectories)
    //         {
    //             double x = traj(i);
    //             double y = traj(i + 1);

    //             // flip rows and columns so x = horizontal axis
    //             int index_y = x / Params::nov::discrete_length_x;
    //             int index_x = y / Params::nov::discrete_length_y;

    //             _image_frames[image_num](index_x, index_y) = 1;
    //         }
    //         if (VERBOSE)
    //         std::cout << _image_frames[image_num] << std::endl;
    //     }
    // }

    private:
    // using matrix directly does not work, see above comment at generate_traj, will not stay in mem after assigining
    // Eigen::Matrix<double, Params::random::max_num_random + 1, Params::sim::trajectory_length> trajectories;
    
    Eigen::VectorXd _params;
    // random trajectories + 1 real one
    std::array<Eigen::VectorXf, Params::random::max_num_random + 1> _trajectories;
    std::array<Eigen::VectorXf, Params::random::max_num_random + 1> _undisturbed_trajectories;
    Eigen::VectorXf _full_trajectory;
    bool _apply_force;
    int _ball_moved_by_noise{-1};
    Eigen::VectorXf _image;
    Eigen::VectorXf _undisturbed_image;
    
    // std::array<Eigen::Matrix<float, Params::nov::discretisation, Params::nov::discretisation>, Params::sim::trajectory_length> _image_frames;
    float _m_entropy;

    bool _observations_generated{false};
    bool _undisturbed_trajectories_generated{false};
};

#endif //TRAJECTORY_HPP
