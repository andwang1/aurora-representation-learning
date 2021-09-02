//
// Created by Andy Wang
//

#ifndef PARAMS_HPP
#define PARAMS_HPP

using namespace sferes::gen::evo_float;

struct Params {
    struct sim {
    SFERES_CONST float ROOM_H = 5.f;
    SFERES_CONST float ROOM_W = 5.f;
    SFERES_CONST float start_x = 3.55f;
    SFERES_CONST float start_y = 3.f;

    SFERES_CONST int trajectory_length = 50;
    SFERES_CONST int num_trajectory_elements = 2 * trajectory_length;

    SFERES_CONST float radius = 0.25f;
    SFERES_CONST bool enable_graphics = false;
    SFERES_CONST float max_force = 5.f;
    SFERES_CONST float sim_duration = 10.f;
    // *100 timesteps in simulation, *2 two coordinates for each timestep
    SFERES_CONST int full_trajectory_length = sim_duration * 100 * 2;
    };

    struct random {
    static double pct_random;
    SFERES_CONST size_t max_num_random = 1;
    };

    struct ae {
    SFERES_CONST size_t batch_size = 256;
    SFERES_CONST size_t nb_epochs = 5000;
    SFERES_CONST float learning_rate = 1e-4;
    SFERES_CONST float CV_fraction = 0.80;

    SFERES_CONST size_t running_mean_num_epochs = 5;
    SFERES_CONST size_t min_num_epochs = 100;

    static double pct_extension;
    static bool full_loss;
    static bool sigmoid;
    static size_t beta;

    enum class loss : unsigned int {Huber, L1, L2, BCE};
    static loss loss_function;

    SFERES_CONST float target_perplexity = batch_size * 0.1;

    // aurora num filter maps
    SFERES_CONST size_t aurora_en_dim1 = 10;
    SFERES_CONST size_t aurora_en_dim2 = 20;
    SFERES_CONST size_t aurora_en_dim3 = 30;

    // network neurons, fully connected
    // input = qd::gen_dim
    SFERES_CONST size_t en_hid_dim1 = 10;
    SFERES_CONST size_t en_hid_dim2 = 20;
    // latent_dim = qd::behav_dim
    
    // filter maps
    SFERES_CONST size_t de_hid_dim1 = 40;
    SFERES_CONST size_t de_hid_dim2 = 20;
    SFERES_CONST size_t de_hid_dim3 = 10;
    // output_dim = sim::trajectory_length

    enum class sne : unsigned int {NoSNE, SNE, TSNE};
    static sne add_sne_criterion;

    static bool sample_train;
    };

    struct update {
    // used in deciding how often to apply dim reduction (and training)
    SFERES_CONST size_t update_frequency = 10; // -1 means exponentially decaying update frequency, how often update BD etc
    SFERES_CONST size_t update_period = 10;
    };
    

    struct pop {
        SFERES_CONST size_t size = 256;
        static size_t nb_gen;
        SFERES_CONST size_t dump_period = 500;
    };

    struct evo_float {
        SFERES_CONST float mutation_rate = 0.1f;
        SFERES_CONST float cross_rate = 0.1f;
        SFERES_CONST mutation_t mutation_type = polynomial;
        SFERES_CONST cross_over_t cross_over_type = sbx;
        SFERES_CONST float eta_m = 15.0f;
        SFERES_CONST float eta_c = 15.0f;
    };
    struct parameters {
        // for manual sim
        SFERES_CONST double min_dpf = 0;
        SFERES_CONST double max_dpf = 0.3f;

        SFERES_CONST double min_angle = -M_PI;
        SFERES_CONST double max_angle = M_PI;
    };

    struct qd {
        SFERES_CONST size_t gen_dim = 4;
        SFERES_CONST int behav_dim = 2;
        // influences l = targeted size of pop
        SFERES_CONST int resolution = 8000; 
        SFERES_CONST int num_train_archives = 0;
        static bool sample;
    };


    struct nov {
        static std::array<double, qd::num_train_archives + 1> l;
        SFERES_CONST double k = 15;
        SFERES_CONST double eps = 0.1;
        // the discretisation used to create the diversity bin data
        SFERES_CONST size_t discretisation = 40;
        SFERES_CONST double discrete_length_x = double(sim::ROOM_W) / nov::discretisation;
        SFERES_CONST double discrete_length_y = double(sim::ROOM_H) / nov::discretisation;
    };

    struct stat {
        SFERES_CONST size_t save_images = 6000;
        SFERES_CONST size_t save_model = 10000;
        SFERES_CONST size_t save_diversity = 500;
        SFERES_CONST int entropy_discretisation = 10;
        SFERES_CONST double ent_discrete_length_x = double(sim::ROOM_W) / entropy_discretisation;
        SFERES_CONST double ent_discrete_length_y = double(sim::ROOM_H) / entropy_discretisation;
    };
};

std::array<double, Params::qd::num_train_archives + 1> Params::nov::l;

// cmd line args
double Params::random::pct_random;
bool Params::ae::full_loss;
bool Params::ae::sigmoid;
size_t Params::pop::nb_gen;
size_t Params::ae::beta;
double Params::ae::pct_extension;
Params::ae::loss Params::ae::loss_function;
bool Params::ae::sample_train;
bool Params::qd::sample;
Params::ae::sne Params::ae::add_sne_criterion;

#endif //PARAMS_HPP