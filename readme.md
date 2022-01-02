# Summary
CoinDetector0 is an algorithm that combines basic image processing steps of smoothing, recoloring, exposure correction, morphological operations and edge and circle detection to identify coins in top-down images of sets of coins. The only user input needed is a path to an image file
# Installation
CoinDetector0 was built for an x64 processor using a Visual Studio 2019 project named HelloOpenCV. Paths to the OpenCV include directories and libraries are set in the .sln and .vcxproj files. The executable file has been built in the x64/debug subfolder of the project directory.   
cmake
a cmake file was generated from the .sln and .vcxproj files using cmake-converter. By: putting CoinDetector0.cpp in a directory, using the cmake . and cmake –build steps, the user can build CoinDetector0 from source. 
Necessary edits to CMakeLists.txt
Set PATH variables `LOCAL_OPENCV_INCLUDE_DIR` and `LOCAL_OPENCV_LIB_DIR` to your system’s paths to the OpenCV header files and libraries
 
Other variables in CMakeLists.txt involve the build platform and will need to be edited if you do not run Windows x64, e.g. `CMAKE_VS_PLATFORM_NAME`.
# Running CoinDetector0
USAGE: `> CoinDetector0(.exe) \[imagefile\]`
Where
`\[imagefile\]` is a path to a plain text file giving the file path for each image in a set to evaluate, one path per line
 OR the path to a single image file, i.e. myimage.jpg