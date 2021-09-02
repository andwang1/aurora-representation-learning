//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_END_POSITIONS_HPP
#define SFERES2_STAT_END_POSITIONS_HPP

#include <sferes/stat/stat.hpp>

namespace sferes {
    namespace stat {

        SFERES_STAT(EndPositions, Stat)
        {
        public:
            typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_diversity == 0))// && (ea.gen() > 0)) 
                {
                   std::string prefix = "end_positions_" + boost::lexical_cast<std::string>(ea.gen());
                    _write_locations(prefix, ea);
                }
            }

            template<typename EA>
            void _write_locations(const std::string &prefix, const EA &ea) const {

                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                std::vector<float> ball_1_x_arr(ea.pop().size());
                std::vector<float> ball_1_y_arr(ea.pop().size());
                std::vector<float> ball_2_x_arr(ea.pop().size());
                std::vector<float> ball_2_y_arr(ea.pop().size());

                std::vector<float> undist_ball_1_x_arr(ea.pop().size());
                std::vector<float> undist_ball_1_y_arr(ea.pop().size());
                std::vector<float> undist_ball_2_x_arr(ea.pop().size());
                std::vector<float> undist_ball_2_y_arr(ea.pop().size());

                std::vector<int> idx_ball_moved(ea.pop().size());

                for (int i{0}; i < ea.pop().size(); ++i)
                {
                    float ball_1_x, ball_1_y, ball_2_x, ball_2_y, undist_ball_1_x, undist_ball_1_y, undist_ball_2_x, undist_ball_2_y;
                    ea.pop()[i]->fit().get_end_positions(ball_1_x, ball_1_y, ball_2_x, ball_2_y, undist_ball_1_x, undist_ball_1_y, undist_ball_2_x, undist_ball_2_y);
                    ball_1_x_arr[i] = ball_1_x;
                    ball_1_y_arr[i] = ball_1_y;
                    ball_2_x_arr[i] = ball_2_x;
                    ball_2_y_arr[i] = ball_2_y;
                    undist_ball_1_x_arr[i] = undist_ball_1_x;
                    undist_ball_1_y_arr[i] = undist_ball_1_y;
                    undist_ball_2_x_arr[i] = undist_ball_2_x;
                    undist_ball_2_y_arr[i] = undist_ball_2_y;

                    idx_ball_moved[i] = ea.pop()[i]->fit().get_idx_ball_moved_by_noise();
                }

                std::ofstream ofs(fname.c_str());
                ofs.precision(8);

                ofs << "Lower Ball X, Lower Ball Y, Higher Ball X, Higher Ball Y, Undisturbed Lower Ball X, Undisturbed Lower Ball Y, Undisturbed Higher Ball X, Undisturbed Higher Ball Y, Index of Ball that was affected by noise\n";

                for (float &i: ball_1_x_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i: ball_1_y_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i: ball_2_x_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i: ball_2_y_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i: undist_ball_1_x_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i: undist_ball_1_y_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i: undist_ball_2_x_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i: undist_ball_2_y_arr)
                    {ofs << i << ",";}
                ofs << "\n";                    
                for (int &i: idx_ball_moved)
                    {ofs << i << ",";}
            }
        };
    }
}

#endif
