#ifndef JERI_TEI_H
#define JERI_TEI_H

#include <libint2.hpp>
#include <jlcxx/jlcxx.hpp>
//#include <jlcxx/stl.hpp>

//#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <limits>
#include <vector>

typedef int64_t julia_int;

//------------------------------------------------------------------------//
//-- C++ JERI engine: small wrapper allowing LibInt to be used in Julia --//
//-- Two Electron Integral Engine: Wrapper class for calls to ------------//
//-- libint2::Engine m_coulomb_eng.compute2 ------------------------------//
//-- Base class defining the interface for Two Electron integral engines--//
//------------------------------------------------------------------------//

class TEIEngine
{
public:
  const libint2::BasisSet *m_basis_set;
  const libint2::ShellPair *m_shellpair_data;

  libint2::Engine m_coulomb_eng;
  TEIEngine(const libint2::BasisSet &t_basis_set,
            const std::vector<libint2::ShellPair> &t_shellpair_data, const int max_nprim, const int max_l)
      : m_basis_set(&t_basis_set),
        m_shellpair_data(t_shellpair_data.data()),
        m_coulomb_eng(libint2::Operator::coulomb,
                      max_nprim,
                      max_l,
                      0)
  {
    m_coulomb_eng.set_precision(0.0); // no screening in the c++ engine
  };

  ~TEIEngine(){};
 };

//------------------------------------------------------------------------//
//-- Two electron integral engine for Restricted Hartree Fock ------------//
//------------------------------------------------------------------------//

class RHFTEIEngine : public TEIEngine
{

public:
  //-- ctors and dtors --//
  RHFTEIEngine(const libint2::BasisSet &t_basis_set,
               const std::vector<libint2::ShellPair> &t_shellpair_data)
      : TEIEngine(t_basis_set, t_shellpair_data, (int)t_basis_set.max_nprim(), (int)t_basis_set.max_l()){};

  ~RHFTEIEngine(){};

  //-- member functions --//
  inline bool compute_eri_block(jlcxx::ArrayRef<double> eri_block,
                                julia_int ash, julia_int bsh, julia_int csh, julia_int dsh,
                                julia_int bra_idx, julia_int ket_idx,
                                julia_int absize, julia_int cdsize)
  {
    m_coulomb_eng.compute2<libint2::Operator::coulomb,
                           libint2::BraKet::xx_xx, 0>((*m_basis_set)[ash - 1], (*m_basis_set)[bsh - 1],
                                                      (*m_basis_set)[csh - 1], (*m_basis_set)[dsh - 1],
                                                      &m_shellpair_data[bra_idx - 1], &m_shellpair_data[ket_idx - 1]);

    //assert(m_coulomb_eng.results()[0] != nullptr);
    if (m_coulomb_eng.results()[0] != nullptr)
    {
      memcpy(eri_block.data(), m_coulomb_eng.results()[0],
             absize * cdsize * sizeof(double));

      return false;
    }
    else
    {
      return true;
    }
  }
};

//------------------------------------------------------------------------//
//-- Two electron integral engine for Density Fitted ---------------------//
//-- Restricted Hartree Fock ---------------------------------------------//
//------------------------------------------------------------------------//

class DFRHFTEIEngine : public TEIEngine
{

public:
  const libint2::BasisSet *m_auxillary_basis_set;
  const libint2::ShellPair *m_auxillary_shellpair_data;
  libint2::Engine m_two_center_engine;

  //-- ctors and dtors --//
  DFRHFTEIEngine(
      const libint2::BasisSet &t_basis_set,
      const libint2::BasisSet &t_auxillary_basis_set,
      const std::vector<libint2::ShellPair> &t_shellpair_data,
      const std::vector<libint2::ShellPair> &t_auxillary_shellpair_data)
      : TEIEngine(t_basis_set, t_shellpair_data, (int)std::max(t_basis_set.max_nprim(), t_auxillary_basis_set.max_nprim()),
                  (int)std::max(t_basis_set.max_l(), t_auxillary_basis_set.max_l()))
  {
    m_two_center_engine = libint2::Engine(libint2::Operator::coulomb,
                                          t_auxillary_basis_set.max_nprim(), t_auxillary_basis_set.max_l(), 0);
    m_auxillary_basis_set = &t_auxillary_basis_set;
    m_auxillary_shellpair_data = t_auxillary_shellpair_data.data();
  }

  ~DFRHFTEIEngine(){};

  //-- member functions --//

  //-- compute_eri_block --//
  //-- ash, csh, dsh, are shell indicies --//
  inline bool compute_eri_block_df(jlcxx::ArrayRef<double> eri_block, julia_int ash, 
                                julia_int csh, julia_int dsh,
                                julia_int copy_size, julia_int memory_skip)
  {
    auto unitShell = libint2::Shell::unit();
    m_coulomb_eng.compute2<libint2::Operator::coulomb,
                           libint2::BraKet::xs_xx, 0>((*m_auxillary_basis_set)[ash - 1], unitShell,
                                                      (*m_basis_set)[csh - 1], (*m_basis_set)[dsh - 1]);
    if (m_coulomb_eng.results()[0] != nullptr)
    {     
      std::copy(m_coulomb_eng.results()[0], m_coulomb_eng.results()[0] + copy_size, eri_block.data() + memory_skip);    
      return false;
    }
    else
    {
      return true;
    }
  }

//-- compute_eri_block --//
  //-- ash, csh, dsh, are shell indicies --//
  inline bool compute_eri_block_two_center_df(jlcxx::ArrayRef<double> eri_block,
                                julia_int shell_1_index, 
                                julia_int shell_2_index,
                                julia_int copy_size, 
                                julia_int memory_skip)
  {
    m_two_center_engine.compute((*m_auxillary_basis_set)[shell_1_index], (*m_auxillary_basis_set)[shell_2_index]);

    if (m_coulomb_eng.results()[0] != nullptr)
    {     
      std::copy(m_coulomb_eng.results()[0], m_coulomb_eng.results()[0] + copy_size, eri_block.data() + memory_skip);    
      return false;
    }
    else
    {
      return true;
    }
  }


  inline bool compute_two_center_eri_block(jlcxx::ArrayRef<double> eri_block, 
  julia_int shell_1_index,
  julia_int shell_2_index,
  julia_int shell_1_basis_count,
  julia_int shell_2_basis_count)
  {    
    auto shell2bf = (*m_auxillary_basis_set).shell2bf();
    int n_basis_functions  =(*m_auxillary_basis_set).nbf();
    m_two_center_engine.compute((*m_auxillary_basis_set)[shell_1_index], (*m_auxillary_basis_set)[shell_2_index]);
    if (m_two_center_engine.results()[0] != nullptr)
    { 
      for(int i = 0; i < shell_1_basis_count* shell_2_basis_count; i++)
      {
          eri_block.data()[i] = *(m_two_center_engine.results()[0] + i);
      }
      return false;
    }
    else
    {
      return true;
    }
  }
};

#endif /* JERI_TEI_H */
