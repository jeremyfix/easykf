#include <boost/python.hpp>
#include "easykf.h"

using namespace boost::python;

namespace easykf {

  namespace bindings {

  }
}


BOOST_PYTHON_MODULE(libpyeasykf)
{
  ////////////////
  // ekf_types.h

  class_<ekf::EvolutionAnneal>("ekf_EvolutionAnneal", init<double, double, double>())
    .def("updateEvolutionNoise", &ekf::EvolutionAnneal::updateEvolutionNoise, "")
    ;

  // Bindings for ekf.h
  // struct ekf::ekf_param, ekf::ekf_state
  class_<ekf::ekf_param>("ekf_param", init<>())
    .def_readwrite("evolution_noise", &ekf::ekf_param::evolution_noise)
    .def_readwrite("observation_noise", &ekf::ekf_param::observation_noise)
    .def_readwrite("n", &ekf::ekf_param::n)
    .def_readwrite("no", &ekf::ekf_param::no)
    .def_readwrite("observation_gradient_is_diagonal", &ekf::ekf_param::observation_gradient_is_diagonal)
    ;

  class_<ekf::ekf_state>("ekf_state", init<>())
    .def_readwrite("xk", &ekf::ekf_state::xk)
    .def_readwrite("xkm", &ekf::ekf_state::xkm)
    .def_readwrite("Pxk", &ekf::ekf_state::Pxk)

    ;


  // def("ekf_param", ekf::ekf_param);
  // def("ekf_state", ekf::ekf_state);

  def("ekf_init", ekf::ekf_init);
  def("ekf_free", ekf::ekf_free);
  //def("ekf_iterate", ekf::ekf_iterate);


  def("ukf_math_min", ukf::math::min);
  def("ukf_math_choleskyUpdate", ukf::math::choleskyUpdate);
	

}
