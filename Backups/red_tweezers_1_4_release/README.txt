RED TWEEZERS 1.4 README
--
This program contains a LabVIEW interface and a C++ OpenGL rendering engine, for control of holographic optical tweezers.  More information is given in the manual.  This file describes the layout of this distribution:

red_tweezers_interface.vi: the top-level LabVIEW interface file.
hologram_engine.exe: the C++ OpenGL rendering engine
hologram_engine_64.exe: as above, compiled for 64-bit systems.
red_tweezers_manual.pdf: manual for the program, in PDF format
red_tweezers_settings.xml: sample settings file for the program
README.txt: this file

Other files are organised into folders:
plugins/  contains plug-ins as described in the manual
hologram_engine_source_code/  contains all C++ and project files to build the hologram engine from source, using Microsoft Visual Studio 2010
subvis/  contains the dependencies of the LabVIEW script, including DLLs used for particle tracking.
extras/  contains some utility VIs, such as an external "abort button" in case the interface crashes, and analysis code for Shack-Hartmann holograms to help set up the system.