#ifndef STANR_RUN_PATHFINDER
#define STANR_RUN_PATHFINDER

#include <cpp11.hpp>
#include <stanr/cpp11_tuple_interop.hpp>
#include <stan/services/pathfinder/single.hpp>
#include <stan/services/pathfinder/multi.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <type_traits>
#include <vector>
#include "r_output.hpp"
#include "r_interrupt.hpp"
#include "r_logger.hpp"
#include "r_worker.hpp"
#include <stanr/r_data_context.hpp>

namespace stanr {
  // A structured_writer that discards everything, including the std::size_t
  // values Pathfinder writes. Stan 2.40's integer overloads are fixed-width,
  // so where std::size_t is neither uint32_t nor uint64_t (macOS, wasm) the
  // base-class call is ambiguous.
  class quiet_structured_writer : public stan::callbacks::structured_writer {
   public:
    using stan::callbacks::structured_writer::write;
    template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    void write(const std::string&, T) {}
  };

  template <class Model>
  cpp11::writable::list run_pathfinder(Model& model, cpp11::list args) {
    const unsigned int seed = stanr::as_cpp<unsigned int>(args["seed"]);
    const unsigned int chain_id = stanr::as_cpp<unsigned int>(args["id"]);
    const double init_radius = stanr::as_cpp<double>(args["init_radius"]);
    const int max_lbfgs_iters = stanr::as_cpp<int>(args["max_lbfgs_iters"]);
    const int history_size = stanr::as_cpp<int>(args["history_size"]);
    const int num_elbo_draws = stanr::as_cpp<int>(args["num_elbo_draws"]);
    const int num_draws = stanr::as_cpp<int>(args["num_draws"]);
    const int num_paths = stanr::as_cpp<int>(args["num_paths"]);
    const int num_psis_draws = stanr::as_cpp<int>(args["num_psis_draws"]);
    const double init_alpha = stanr::as_cpp<double>(args["init_alpha"]);
    const double tol_obj = stanr::as_cpp<double>(args["tol_obj"]);
    const double tol_rel_obj = stanr::as_cpp<double>(args["tol_rel_obj"]);
    const double tol_grad = stanr::as_cpp<double>(args["tol_grad"]);
    const double tol_rel_grad = stanr::as_cpp<double>(args["tol_rel_grad"]);
    const double tol_param = stanr::as_cpp<double>(args["tol_param"]);
    const bool save_single_paths = stanr::as_cpp<bool>(args["save_single_paths"]);
    const bool psis_resample = stanr::as_cpp<bool>(args["psis_resample"]);
    const bool calculate_lp = stanr::as_cpp<bool>(args["calculate_lp"]);
    const int refresh = stanr::as_cpp<int>(args["refresh"]);
    const bool verbose = stanr::as_cpp<bool>(args["verbose"]);
    const bool show_exceptions = stanr::as_cpp<bool>(args["show_exceptions"]);

    cpp11::list init_list = args["init"];

    stanr::r_logger logger(verbose, show_exceptions);
    // Single-path stays on the R thread; multi-path uses a coordinator
    // thread, so it uses an atomic interrupt.
    stanr::r_interrupt interrupt(num_paths <= 1);

    int return_code;

    if (num_paths <= 1) {
      // Single pathfinder
      stanr::r_data_context init_ctx(init_list, args["init_declarations"]);
      stan::callbacks::writer init_writer;
      stanr::r_sample_writer sample_writer(num_draws);
      // No-op: the LBFGS inverse-metric estimate is not exposed to R.
      stanr::quiet_structured_writer metric_writer;

      return_code = stan::services::pathfinder::pathfinder_lbfgs_single<false, false>(
          model, init_ctx, seed, chain_id, init_radius,
          history_size, init_alpha,
          tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
          max_lbfgs_iters, num_elbo_draws, num_draws, save_single_paths,
          refresh,
          interrupt, logger,
          init_writer, sample_writer,
          metric_writer,
          calculate_lp);

      logger.flush();
      cpp11::writable::strings output = cpp11::as_sexp(logger.history());
      return cpp11::writable::list({
        cpp11::named_arg("return_code") = return_code,
        cpp11::named_arg("draws") = sample_writer.to_r_matrix(),
        cpp11::named_arg("output") = output
      });
    } else {
      // Multi-path runs in a coordinator std::thread; all contexts/writers
      // are C++ owned before launch. The immutable context is shared.
      const auto init_ctx = std::make_unique<stanr::r_data_context>(
          init_list, args["init_declarations"]);
      std::vector<stan::io::var_context*> init_ctxs(num_paths, init_ctx.get());

      // Only the PSIS-resampled draws are returned; base writers are no-ops
      // so Stan doesn't retain every per-path candidate draw.
      std::vector<stan::callbacks::writer> single_param_writers(num_paths);
      std::vector<stanr::quiet_structured_writer> single_diag_writers(
          num_paths);

      // Final combined parameter writer
      stanr::r_sample_writer param_writer(num_psis_draws);
      stanr::quiet_structured_writer diag_writer;

      // Init writers (one per path, for writing initial values)
      std::vector<stan::callbacks::writer> init_writers(num_paths);

      return_code = stanr::run_on_worker_thread(
          logger, "Pathfinder",
          [&](stanr::r_interrupt& worker_interrupt) -> int {
            return stan::services::pathfinder::pathfinder_lbfgs_multi(
                model, std::move(init_ctxs), seed, chain_id, init_radius,
                history_size, init_alpha,
                tol_obj, tol_rel_obj, tol_grad, tol_rel_grad, tol_param,
                max_lbfgs_iters, num_elbo_draws, num_draws, num_psis_draws,
                num_paths, save_single_paths, refresh, worker_interrupt, logger,
                init_writers, single_param_writers, single_diag_writers,
                param_writer, diag_writer, calculate_lp, psis_resample);
          });

      cpp11::writable::strings output = cpp11::as_sexp(logger.history());
      return cpp11::writable::list({
        cpp11::named_arg("return_code") = return_code,
        cpp11::named_arg("draws") = param_writer.to_r_matrix(),
        cpp11::named_arg("output") = output
      });
    }
  }
}

#endif
