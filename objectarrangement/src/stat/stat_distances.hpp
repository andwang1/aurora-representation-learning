//
// Created by Andy Wang
//

#ifndef SFERES2_STAT_DISTANCES_HPP
#define SFERES2_STAT_DISTANCES_HPP

#include <sferes/stat/stat.hpp>
#include <numeric>

namespace sferes {
    namespace stat {

        SFERES_STAT(Distances, Stat)
        {
        public:
            Distances(){}

            template<typename EA>
            void refresh(EA &ea) 
            {
                if ((ea.gen() % Params::stat::save_diversity == 0)) 
                {
                   std::string prefix = "distances" + boost::lexical_cast<std::string>(ea.gen());
                    _write_distances(prefix, ea);
                }
            }

            template<typename EA>
            void _write_distances(const std::string &prefix, const EA &ea) 
            {
                std::string fname = ea.res_dir() + "/" + prefix + std::string(".dat");
                std::cout << "writing... " << fname << std::endl;

                std::vector<float> distances1_arr(ea.pop().size());
                std::vector<float> distances2_arr(ea.pop().size());
                std::vector<float> undist_distances1_arr(ea.pop().size());
                std::vector<float> undist_distances2_arr(ea.pop().size());


                for (int i{0}; i < ea.pop().size(); ++i)
                {
                    float distances1, distances2, undist_distances1, undist_distances2;
                    ea.pop()[i]->fit().calculate_distance(distances1, distances2, undist_distances1, undist_distances2);
                    distances1_arr[i] = distances1;
                    distances2_arr[i] = distances2;
                    undist_distances1_arr[i] = undist_distances1;
                    undist_distances2_arr[i] = undist_distances2;
                }
                
                std::ofstream ofs(fname.c_str());
                ofs.precision(8);
                ofs << "Lower Ball Distance, Higher Ball Distance, Undisturbed Lower Ball Distance, Undisturbed Higher Ball Distance\n";
                
                for (float &i : distances1_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i : distances2_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i : undist_distances1_arr)
                    {ofs << i << ",";}
                ofs << "\n";
                for (float &i : undist_distances2_arr)
                    {ofs << i << ",";}
            }
        };
    }
}


#endif
