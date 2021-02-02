import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt

import os
import time
import random

import constants as cst
import classes as cls

def execute (ver):
   cnt_met = 0
   cnt_chan = 0
   lst_lst_err = [] # each s_g, each method
   lst_lst_time = [] # each s_g, each method
   lst_met = cst.LST_MET ()

   lst_chan_met = np.array (list (map (cst.NUM_CHAN_MET, lst_met)))
   num_chan_tot = cst.NUM_S_G () * lst_chan_met.sum ()

   time_tot_start = time.time ()
   for i_s_g in range (cst.NUM_S_G ()):
      s_g = cst.S_G_INIT () * (cst.SCALE_S_G () ** i_s_g)
      print ("σ = ", '%.2f' % s_g, " :", sep = '')
      lst_err = [0] * cst.NUM_MET ()
      lst_time = [0] * cst.NUM_MET ()
      for j_met in range (cst.NUM_MET ()):
         met = lst_met [j_met]
         cnt_met += 1
         num_chan_met = cst.NUM_CHAN_MET (met)

         for _ in range (num_chan_met):
            print ("channel instance ", cnt_chan, "...", sep = '', end = '\r', flush = True)
            kk = get_kk (ver)
            cnt_chan += 1
            hh = pick_hh (ver)
            gg = kk.conj ().T @ hh @ kk
            g = vectorize (gg)
            norm_hh = np.linalg.norm (hh, ord = 'fro')
            g_r_h = np.zeros (2 * cst.NN_H (ver))

            y_tot = None
            pp_tot = None
            time_chan_start = time.time ()
            for _ in range (cst.NUM_STAGE (ver)):
               ff_bb = pick_mat_bb (cst.NN_YY_t (ver), ver)
               ff_rr = pick_mat_rr (cst.NN_YY_t (ver), ver).T
               ww_bb = pick_mat_bb (cst.NN_YY_r (ver), ver)
               ww_rr = pick_mat_rr (cst.NN_YY_r (ver), ver)
               pp = np.kron (ff_bb.T @ ff_rr.T @ kk.conj (), ww_bb @ ww_rr @ kk)
               zz = pick_zz (ver)
               z = vectorize (zz)
               y = pp @ g + (s_g / np.sqrt(2)) * z

               y_r = find_rep_vec (y)
               pp_r = find_rep_mat (pp)

               if y_tot is not None:
                  y_tot = np.concatenate ((y_tot, y_r), axis = 0)
                  pp_tot = np.concatenate ((pp_tot, pp_r), axis = 0)
               else:
                  y_tot = y_r
                  pp_tot = pp_r

            if (met == cls.Method.LLSS):
               g_r_h = llss (pp_tot, y_tot, ver)
            elif (met == cls.Method.OOMMPP_TWO):
               g_r_h = oommpp_two (pp_tot, y_tot, cst.H_G_OOMMPP_TWO (ver) * s_g, ver)
            elif (met == cls.Method.OOMMPP_INFTY):
               g_r_h = oommpp_infty (pp_tot, y_tot, cst.H_G_OOMMPP_INFTY (ver) * s_g, ver)
            elif (met == cls.Method.LASSO):
               g_r_h = lasso_qqpp (pp_tot, y_tot, cst.G_G_LASSO (ver) * s_g, ver)
            elif (met == cls.Method.DDSS):
               g_r_h = ddss_llpp (pp_tot, y_tot, cst.G_G_DDSS (ver) * s_g, ver)
            if (np.linalg.norm (g_r_h, ord = 2) > cst.MAX_NORM (ver)):
               g_r_h = np.random.normal (0, 1, 2 * cst.NN_H (ver))

            rr = error_norm (hh, g_r_h, s_g, ver)
            lst_err [j_met] += rr / num_chan_met
            time_chan_stop = time.time ()
            lst_time [j_met] += (time_chan_stop - time_chan_start) / (60 * num_chan_met)
         time_hold = time.time ()

         progress = cnt_chan / num_chan_tot
         print ("   method ", cnt_met, "                    ", sep = '', end = '\n', flush = True)
         print ("     (", '%.1f' % (100 * progress), "%; ",
               '%.2f' % ((time_hold - time_tot_start) * (1 - progress) / (progress * 60)), " min. remaining)",
               sep = '', flush = True)
         print ("                            ", end = '\r') # clear last line
         print ("   done")

      lst_lst_err.append (lst_err)
      lst_lst_time.append (lst_time)
   time_tot_stop = time.time ()

   print (cnt_chan, "channel instances simulated")
   print (
         "total time elapsed: ",
         '%.2f' % ((time_tot_stop - time_tot_start) / 60),
         " (min)", flush = True)

   arr_s_g = (cst.S_G_INIT ()
         * cst.SCALE_S_G () ** (np.array (range (cst.NUM_S_G ()))))
   arr_x = -10 * np.array (np.log (arr_s_g)) / np.log (10)

   lst_lst_err = list (np.array (lst_lst_err).T) # each method, each s_g
   lst_arr_err = [10 * np.array (np.log (lst)) / np.log (10) for lst in lst_lst_err]
   label_x = "Signal level (dB)"
   label_y = "Relative error (dB)"
   save_table (arr_x, lst_arr_err,
      label_x, label_y, cst.LEGEND (),
      "error", ver)
   save_plot (arr_x, lst_arr_err,
      label_x, label_y, cst.LEGEND (),
      "error", ver)

   lst_lst_time = list (np.array (lst_lst_time).T) # each meth, each s_g
   lst_arr_time = [np.array (lst) for lst in lst_lst_time]
   label_x = "Signal level (dB)"
   label_y = "Time in minutes"
   save_table (
      arr_x, lst_arr_time,
      label_x, label_y, cst.LEGEND (),
      "time", ver)
   save_plot (
      arr_x, lst_arr_time,
      label_x, label_y, cst.LEGEND (),
      "time", ver)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def llss (pp_r, y_r, ver):
   try:
      pp_r_inv = np.linalg.pinv (pp_r)
   except (np.linalg.LinAlgError) as err:
      print ("Moore Penrose inverse does not exist!", flush = True)
      print (err)
      g_r_h = np.zeros (pp_r.shape [1])
      return g_r_h

   g_r_h = pp_r_inv @ y_r
   return g_r_h

def lasso_qqpp (pp_r, y_r, g_g, ver):
   nn = pp_r.shape [1]
   k = - 2 * pp_r.T @ y_r
   qq = pp_r.T @ pp_r
   g_r = cp.Variable ((nn))
   l_g = cst.NN_HH (ver) / g_g

   prob = cp.Problem (
         cp.Minimize (
               cp.norm (pp_r @ g_r - y_r, 2) ** 2
               + l_g * cp.norm (g_r, 1)))

   try:
      prob.solve (solver = cp.ECOS)
      g_r_h = g_r.value
   except (cp.error.SolverError, cp.error.DCPError) as err:
      print ("Lasso fails to solve the convex program!", flush = True)
      print (err)
      g_r_h = np.linalg.pinv (pp_r) @ y_r
   return g_r_h

def ddss_llpp (pp_r, y_r, g_g, ver):
   nn = pp_r.shape [1]
   g_r = cp.Variable (nn)
   g_r_abs = cp.Variable (nn)
   k = pp_r.T @ y_r
   qq = pp_r.T @ pp_r
   c = np.ones ((nn))
   l_g = cst.NN_HH (ver) / g_g

   prob = cp.Problem (
         cp.Minimize (
            cp.norm (g_r, 1)
            + l_g * cp.norm (pp_r.T @ (y_r - pp_r @ g_r), 'inf')))

   try:
      prob.solve (solver = cp.ECOS)
      g_r_h = g_r.value
   except (cp.error.SolverError, cp.error.DCPError) as err:
      print ("Dantzig Selector fails to solve the convex program!", flush = True)
      print (err)
      g_r_h = np.linalg.pinv (pp_r) @ y_r
   return g_r_h

def oommpp_two (pp_r, y_r, h_g, ver):
   nn = pp_r.shape [1]
   r = y_r # remained vector
   tt = range (nn) # list of column indices
   ss = [] # estimated nonzero components
   cnt_iter = 0
   while True:
      cnt_iter += 1
      lst_match = [abs (pp_r [:, i].T @ r) for i in tt]
      s = np.argmax (lst_match)
      ss.append (s)
      ss = list (sorted (set (ss)))
      pp_r_ss = extract_mat (pp_r, ss)
      pp_r_ss_inv = np.linalg.pinv (pp_r_ss)

      r = y_r - pp_r_ss @ pp_r_ss_inv @ y_r
      if (np.linalg.norm (r, ord = 2) <= h_g
            or (cnt_iter >= cst.H_G_OOMMPP_TWO (ver))):
         break
   g_r_ss_h = pp_r_ss_inv @ y_r
   g_r_h = np.zeros (nn)
   embed_subvec (g_r_h, ss, g_r_ss_h)
   return g_r_h

def oommpp_infty (pp_r, y_r, h_g, ver):
   nn = pp_r.shape [1]
   r = y_r # remainder
   tt = range (nn) # list of column indices
   ss = [] # extracted column indices
   cnt_iter = 0
   while True:
      cnt_iter += 1
      lst_match = [abs (pp_r [:, i].T @ r) for i in tt]
      s = np.argmax (lst_match)
      ss.append (s)
      ss = list (sorted (set (ss)))
      pp_r_ss = extract_mat (pp_r, ss)
      pp_r_ss_inv = np.linalg.pinv (pp_r_ss)

      r = y_r - pp_r_ss @ pp_r_ss_inv @ y_r
      if (np.linalg.norm (pp_r.T @ r, ord = np.inf) <= h_g
            or cnt_iter >= cst.H_G_OOMMPP_INFTY (ver)):
         break
   g_r_ss_h = pp_r_ss_inv @ y_r
   g_r_h = np.zeros (nn)
   embed_subvec (g_r_h, ss, g_r_ss_h)
   return g_r_h

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def mat_complex_normal (nn_1, nn_2):
   return ((np.random.normal (0, 1, (nn_1, nn_2))
      + 1J * np.random.normal (0, 1, (nn_1, nn_2))))

def pick_zz (ver):
   return mat_complex_normal (cst.NN_YY_t (ver), cst.NN_YY_r (ver))

def pick_mat_bb (nn, ver):
   return (mat_complex_normal (nn, nn) / np.sqrt (nn))

def pick_mat_rr (nn, ver):
   kk = np.sqrt (cst.NN_HH (ver)) * get_kk (ver)
   kk_ss = kk [random.sample (list (range (cst.NN_HH (ver))), nn), :]
   return kk_ss

def pick_hh (ver):
   m = np.random.uniform (0, 2 * np.pi)
   ret = np.zeros ((cst.NN_HH (ver), cst.NN_HH (ver)), dtype = complex)
   for _ in range (cst.LL (ver)):
      alpha = (np.random.normal (0, cst.NN_HH (ver) / cst.LL (ver))
         + 1J * np.random.normal (0, cst.NN_HH (ver) / cst.LL (ver)))
      phi = (2 * np.pi * (cst.DIST_ANT (ver) /cst.LAMBDA_ANT (ver))
         * np.sin (np.random.uniform (m, 2 * np.pi /24)))
      theta = (2 * np.pi * (cst.DIST_ANT (ver) /cst.LAMBDA_ANT (ver))
         * np.sin (np.random.uniform (m, 2 * np.pi /24)))
      ret += alpha * np.outer (arr_resp (phi, ver), arr_resp (theta, ver))
   return ret

def get_kk (ver):
   ret = np.zeros ((cst.NN_HH (ver), cst.NN_HH (ver)), dtype = complex)
   for n_1 in range (cst.NN_HH (ver)):
      for n_2 in range (cst.NN_HH (ver)):
         ret [n_1] [n_2] = ((1 / np.sqrt (cst.NN_HH (ver)))
            * np.exp (2 * np.pi * 1J * n_1 * n_2 / cst.NN_HH (ver)))
   return ret

def arr_resp (t, ver):
   return ((1 / np.sqrt (cst.NN_HH (ver)))
      * np.array ([np.exp (1J * i * t) for i in range (cst.NN_HH (ver))]))

def error_norm (hh, g_r_h, s_g, ver):
   kk = get_kk (ver)
   g_h = inv_find_rep_vec (g_r_h)
   gg_h = inv_vectorize (g_h, cst.NN_HH (ver), cst.NN_HH (ver))
   hh_h = kk @ gg_h @ kk.conj().T
   nor_hh = np.linalg.norm (hh, ord = 'fro')
   nor_ee = np.linalg.norm (hh - hh_h, ord = 'fro')
   #return (nor_ee + s_g) / nor_hh
   return nor_ee / nor_hh

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def find_rep_vec (v):
   ret =np.zeros ((2 * len (v)))
   for i in range (len (v)):
      ret [2 * i] = np.real (v [i])
      ret [2 * i + 1] = np.imag (v [i])
   return ret

def inv_find_rep_vec (v):
   assert (len (v) % 2 == 0)
   len_v =int (len (v) / 2)
   v_re =np.array ([v [2 * i] for i in range (len_v)])
   v_im =np.array ([v [2 * i + 1] for i in range (len_v)])
   return v_re +1J *v_im

def find_rep_mat (aa):
   ret =np.zeros ((2 * (aa.shape[0]), 2 * (aa.shape[1])))
   for i in range (aa.shape[0]):
      for j in range (aa.shape[1]):
         ret [2 * i] [2 * j] = np.real (aa [i, j])
         ret [2 * i + 1] [2 * j] = np.imag (aa [i, j])
         ret [2 * i] [2 * j + 1] = -np.imag (aa [i, j])
         ret [2 * i + 1] [2 * j + 1] = np.real (aa [i, j])
   return ret

def indic_vec (nn, i):
   ret = np.zeros ((nn), dtype = 'bool')
   ret [i] = 1
   return ret

def indic_rep_mat (nn, i):
   ret = np.zeros ((2 * nn, 2 * nn), dtype = 'bool')
   ret [2 * i] [2 * i] = 1
   ret [2 * i + 1] [2 * i + 1] = 1
   return ret

def vectorize (v):
   return np.reshape (v, (1, -1)) [0]

def inv_vectorize (v, nn_1, nn_2):
   assert (len (v) == nn_1 * nn_2)
   return np.reshape (v, (nn_1, -1))

def get_supp (v, sp):
   ss = list (np.sort (np.argsort (np.abs (v)) [-sp:]))
   return ss

def mask_vec (u, ss):
   ll = len (ss)
   v = np.zeros (u.shape)
   for i in range (ll):
      v [ss[i]] = u [ss[i]]
   return v

def extract_mat (uu, ss):
   ll = len (ss)
   vv = np.zeros ((uu.shape [0], ll))
   for i in range (ll):
      vv [:, i] = uu [:, ss[i]]
   return vv

def embed_subvec (u, ss, v):
   ll = len (ss)
   for i in range (ll):
      u [ss[i]] = v [i]

def get_str_ver (ver):
   switcher_size = {
      cls.Size.SMALL: "small",
      cls.Size.MEDIUM: "medium",
      cls.Size.BIG: "big"}
   switcher_ratio = {
      cls.Ratio.TALL: "tall",
      cls.Ratio.WIDE: "wide",
      cls.Ratio.SQUARE: "square"}
   switcher_stage = {
      cls.Stage.ONE: "one",
      cls.Stage.TWO: "two",
      cls.Stage.FOUR: "four"}
   title = (str (get_identity (ver)) + "-" +
         switcher_size [ver.size] + "-" +
         switcher_ratio [ver.ratio] + "-" +
         switcher_stage [ver.stage])
   return title

def get_identity (ver):
   switcher_size = {
      cls.Size.SMALL: 0,
      cls.Size.MEDIUM: 1,
      cls.Size.BIG: 2}
   switcher_ratio = {
      cls.Ratio.TALL: 0,
      cls.Ratio.WIDE: 1,
      cls.Ratio.SQUARE: 2}
   switcher_stage = {
      cls.Stage.ONE: 0,
      cls.Stage.TWO: 1,
      cls.Stage.FOUR: 2}
   iden = (9 * switcher_size [ver.size] +
         3 * switcher_ratio [ver.ratio] +
         switcher_stage [ver.stage])
   return iden

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def save_plot (arr_x, lst_arr_y, label_x, label_y, lst_legend, str_title, ver):

   plt.close ("all")
   plt.xlabel (label_x, fontsize = 12)
   plt.ylabel (label_y, fontsize = 12)

   num_style = 3
   lst_style = ['-', '--', ':']
      # '-', '--', '-.', ':':
      # solid, long dotted, long-short dotted, short dotted
   num_color = 4
   lst_color = ['r', 'g', 'b', 'k']
      # 'r', 'g', 'c', 'b', 'k':
      # red, green, cyan, blue, black
   num_marker = 5
   lst_marker = ['v', '^', 'o', 's', 'D']
      # 'v', '^', 'o', 's', '*', 'D':
      # triangle down, triangle up, circle, square, star, diamond
   size_marker = 6
   width_line = 2

   assert (len (lst_arr_y) == len (lst_legend))
   for i_met in range (len (lst_arr_y)):
      arr_y = lst_arr_y [i_met]
      assert (len (arr_x) == len (arr_y))
      plt.plot (
         arr_x,
         arr_y,
         markersize = size_marker,
         linewidth = width_line,
         linestyle = lst_style [int (i_met % num_style)],
         color = lst_color [int (i_met % num_color)],
         marker = lst_marker [int (i_met % num_marker)],
         label = lst_legend [i_met])
   plt.legend (
      bbox_to_anchor = (1.05, 1),
      loc = 'upper left',
      borderaxespad = 0.)

   os.system ("mkdir -p ../plt")
   path_plot_out = (
      os.path.abspath (os.path.join (os.getcwd (), os.path.pardir))
      + "/plt/" + str_title + "-" + get_str_ver (ver) + ".png")
   if os.path.isfile (path_plot_out):
      os.system ("rm -f " + path_plot_out)
   plt.savefig (path_plot_out, bbox_inches = "tight")
   plt.close ()

def save_table (arr_x, lst_arr_y, label_x, label_y, lst_legend, str_title, ver):
   os.system ("mkdir -p ../dat")
   path_table_out = (
      os.path.abspath (os.path.join (os.getcwd (), os.path.pardir))
      + "/dat/" + str_title + "-" + get_str_ver (ver) + ".txt")
   if os.path.isfile (path_table_out):
      os.system ("rm -f " + path_table_out)

   with open (path_table_out, 'w') as the_file:
      the_file.write (label_x + '\t')
      the_file.write ('\t'.join (map (str, arr_x)) + '\n')
      assert (len (lst_arr_y) == len (lst_legend))
      for i in range (len (lst_arr_y)):
         the_file.write (
            lst_legend [i] + '\t'
               + '\t'.join (map (str, lst_arr_y [i])) + '\n')

