#include "field_factory.hxx"

#include "hermes-2.hxx"
#include "div_ops.hxx"
#include "bout/fv_ops.hxx"

std::vector<std::string> getAll(std::string str) {
  std::vector<std::string> out{};
  int i = 0;
  while (true) {
    std::string section = fmt::format(str, i++);

    output.write("Trying '{}'\n", section);
    //auto sec = Options::root()[section];
    //if (sec.getChildren().empty()){
    if (not Options::root().isSection(section)) {
      return out;
    }
    out.push_back(section);
  }
}

class nameandfunc2 {
public:
  std::string name;
  std::function<Field3D(const Field3D&, const Field3D&)> func;
};

class nameandfunc1 {
public:
  std::string name;
  std::function<Field3D(const Field3D&)> func;
};

const auto functions = {
  nameandfunc2{"R", [] (const Field3D& R, const Field3D& Z) {return R; }},
  nameandfunc2{"R²", [] (const Field3D& R, const Field3D& Z) {return R*R; }},
  nameandfunc2{"sin(R)", [] (const Field3D& R, const Field3D& Z) {return sin(R); }},
  nameandfunc2{"sin(Z)", [] (const Field3D& R, const Field3D& Z) {return sin(Z); }},
  nameandfunc2{"sin(Z)*sin(R)", [] (const Field3D& R, const Field3D& Z) {return sin(Z)*sin(R); }},
};


const auto difops = {
  nameandfunc2{"FV::Div_a_Grad_perp(1, f)", [] (const Field3D& a, const Field3D& f) { return FV::Div_a_Grad_perp(a, f); }},
  nameandfunc2{"FCI::Div_a_Grad_perp(1, f)", [] (const Field3D& a, const Field3D& f) { return FCI::Div_a_Grad_perp(a, f); }},
  nameandfunc2{"Div_a_Grad_perp_nonorthog(1, f)", [] (const Field3D& a, const Field3D& f) { return Div_a_Grad_perp_nonorthog(a, f);}},
  nameandfunc2{"Delp2(f)", [] (const Field3D& a, const Field3D& f) { return Delp2(f); }},
  nameandfunc2{"Laplace(f)", [] (const Field3D& a, const Field3D& f) { return Laplace(f); }},
};

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  auto meshes = getAll("mesh_{}");
  auto fields = getAll("field_{}");
  
  output.write("Found {:d} meshes and {:d} fields", meshes.size(), fields.size());
  for (const auto& meshname: meshes) {
    Mesh* mesh = Mesh::create(&Options::root()[meshname]);
    mesh->load();

    Options dump;
    Field3D R{mesh}, Z{mesh};
    mesh->get(R, "R", 0.0, false);
    mesh->get(Z, "Z", 0.0, false);

    dump["R"] = R;
    dump["Z"] = Z;


    auto coord = mesh->getCoordinates();
    //coord->g23, coord->g_23, coord->dy, coord->dz, coord->Bxy, coord->J)

    
    Field3D a{1.0, mesh};
    
    mesh->communicate(a, coord->g12, coord->g_12, coord->g23, coord->g_23, coord->dy, coord->dz, coord->Bxy, coord->J);
    a.applyParallelBoundary("parallel_neumann_o2");
    coord->g23.applyParallelBoundary("parallel_neumann_o2");
    coord->g_23.applyParallelBoundary("parallel_neumann_o2");
    coord->g12.applyParallelBoundary("parallel_neumann_o2");
    coord->g_12.applyParallelBoundary("parallel_neumann_o2");
    coord->dy.applyParallelBoundary("parallel_neumann_o2");
    coord->dz.applyParallelBoundary("parallel_neumann_o2");
    coord->Bxy.applyParallelBoundary("parallel_neumann_o2");
    coord->J.applyParallelBoundary("parallel_neumann_o2");
    
    int i = 0;
    for (const auto& func: functions) {
      auto f = func.func(R, Z);
      mesh->communicate(f);
      f.applyParallelBoundary("parallel_neumann_o2");
      for (const auto& dif: difops) {
	auto outname = fmt::format("out_{}", i++);
	dump[outname] = dif.func(a, f);
	dump[outname].setAttributes({
	    {"operator", dif.name},
	    {"function", func.name},
	    {"f", func.name}
	  });
      }
    }
    
    if (mesh) {
      mesh->outputVars(dump);
      dump["BOUT_VERSION"].force(bout::version::as_double);
    }

      std::string outname = fmt::format(
          "{}/BOUT.{}.{}.nc",
          Options::root()["datadir"].withDefault<std::string>("data"), meshname, BoutComm::rank());
      
      bout::OptionsNetCDF(outname).write(dump);
      
  };
  
  BoutFinalise()    ;
}
//   std::vector<Field3D> fields;
//   fields.resize(static_cast<int>(BoundaryParType::SIZE));
//   Options dump;
//   for (int i=0; i< fields.size(); i++){
//     fields[i] = Field3D{0.0};
//     mesh->communicate(fields[i]);
//     for (const auto &bndry_par : mesh->getBoundariesPar(static_cast<BoundaryParType>(i))) {
//       output.write("{:s} region\n", toString(static_cast<BoundaryParType>(i)));
//       for (bndry_par->first(); !bndry_par->isDone(); bndry_par->next()) {
//         fields[i][bndry_par->ind()] += 1;
//         output.write("{:s} increment\n", toString(static_cast<BoundaryParType>(i)));
//       }
//     }
//     output.write("{:s} done\n", toString(static_cast<BoundaryParType>(i)));
//     dump[fmt::format("field_{:s}", toString(static_cast<BoundaryParType>(i)))] = fields[i];
//   }

//   bout::writeDefaultOutputFile(dump);

//   BoutFinalise();
// }




