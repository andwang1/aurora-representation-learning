#!/bin/bash
./waf configure --exp imagesd --cpp14=yes --kdtree /workspace/include --robox2d /workspace --magnum_install_dir /workspace --magnum_integration_install_dir /workspace --magnum_plugins_install_dir /workspace --corrade_install_dir /workspace

./waf --exp imagesd -j 1
echo 'FINISHED BUILDING. Now fixing name of files'
python -m fix_build --path-folder /git/sferes2/build/exp/imagesd/
