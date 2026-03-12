#include "hope/reeds_shepp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace hope {

namespace {
constexpr double kStepSize = 0.2;
constexpr double kMaxLength = 1000.0;
constexpr double kPi = 3.14159265358979323846;

struct PathInternal {
    std::vector<double> lengths;
    std::vector<char> ctypes;
    double total_length = 0.0;
};

double M(double theta) {
    double phi = std::fmod(theta, 2.0 * kPi);
    if (phi < -kPi) {
        phi += 2.0 * kPi;
    }
    if (phi > kPi) {
        phi -= 2.0 * kPi;
    }
    return phi;
}

std::pair<double, double> R(double x, double y) {
    return {std::hypot(x, y), std::atan2(y, x)};
}

bool SetPath(std::vector<PathInternal>& paths, const std::vector<double>& lengths, const std::vector<char>& ctypes) {
    PathInternal path;
    path.ctypes = ctypes;
    path.lengths = lengths;

    for (const auto& existing : paths) {
        if (existing.ctypes == path.ctypes) {
            double diff_sum = 0.0;
            for (size_t i = 0; i < existing.lengths.size(); ++i) {
                diff_sum += existing.lengths[i] - path.lengths[i];
            }
            if (diff_sum <= 0.01) {
                return false;
            }
        }
    }

    double total = 0.0;
    for (double l : lengths) {
        total += std::abs(l);
    }
    if (total >= kMaxLength || total < 0.001) {
        return false;
    }

    path.total_length = total;
    paths.push_back(path);
    return true;
}

bool LSL(double x, double y, double phi, double& t, double& u, double& v) {
    auto [u1, t1] = R(x - std::sin(phi), y - 1.0 + std::cos(phi));
    if (t1 >= 0.0) {
        t = t1;
        u = u1;
        v = M(phi - t);
        if (v >= 0.0) {
            return true;
        }
    }
    return false;
}

bool LSR(double x, double y, double phi, double& t, double& u, double& v) {
    auto [u1, t1] = R(x + std::sin(phi), y - 1.0 - std::cos(phi));
    u1 = u1 * u1;
    if (u1 >= 4.0) {
        u = std::sqrt(u1 - 4.0);
        const double theta = std::atan2(2.0, u);
        t = M(t1 + theta);
        v = M(t - phi);
        if (t >= 0.0 && v >= 0.0) {
            return true;
        }
    }
    return false;
}

bool LRL(double x, double y, double phi, double& t, double& u, double& v) {
    auto [u1, t1] = R(x - std::sin(phi), y - 1.0 + std::cos(phi));
    if (u1 <= 4.0) {
        u = -2.0 * std::asin(0.25 * u1);
        t = M(t1 + 0.5 * u + kPi);
        v = M(phi - t + u);
        if (t >= 0.0 && u <= 0.0) {
            return true;
        }
    }
    return false;
}

bool SLS(double x, double y, double phi, double& t, double& u, double& v) {
    phi = M(phi);
    if (y > 0.0 && 0.0 < phi && phi < kPi * 0.99) {
        const double xd = -y / std::tan(phi) + x;
        t = xd - std::tan(phi / 2.0);
        u = phi;
        v = std::sqrt((x - xd) * (x - xd) + y * y) - std::tan(phi / 2.0);
        return true;
    } else if (y < 0.0 && 0.0 < phi && phi < kPi * 0.99) {
        const double xd = -y / std::tan(phi) + x;
        t = xd - std::tan(phi / 2.0);
        u = phi;
        v = -std::sqrt((x - xd) * (x - xd) + y * y) - std::tan(phi / 2.0);
        return true;
    }
    return false;
}

void SCS(double x, double y, double phi, std::vector<PathInternal>& paths) {
    double t = 0.0, u = 0.0, v = 0.0;
    if (SLS(x, y, phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'S', 'L', 'S'});
    }
    if (SLS(x, -y, -phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'S', 'R', 'S'});
    }
}

void CSC(double x, double y, double phi, std::vector<PathInternal>& paths) {
    double t = 0.0, u = 0.0, v = 0.0;
    if (LSL(x, y, phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'L', 'S', 'L'});
    }
    if (LSL(-x, y, -phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'L', 'S', 'L'});
    }
    if (LSL(x, -y, -phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'R', 'S', 'R'});
    }
    if (LSL(-x, -y, phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'R', 'S', 'R'});
    }

    if (LSR(x, y, phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'L', 'S', 'R'});
    }
    if (LSR(-x, y, -phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'L', 'S', 'R'});
    }
    if (LSR(x, -y, -phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'R', 'S', 'L'});
    }
    if (LSR(-x, -y, phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'R', 'S', 'L'});
    }
}

void CCC(double x, double y, double phi, std::vector<PathInternal>& paths) {
    double t = 0.0, u = 0.0, v = 0.0;
    if (LRL(x, y, phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'L', 'R', 'L'});
    }
    if (LRL(-x, y, -phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'L', 'R', 'L'});
    }
    if (LRL(x, -y, -phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'R', 'L', 'R'});
    }
    if (LRL(-x, -y, phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'R', 'L', 'R'});
    }

    const double xb = x * std::cos(phi) + y * std::sin(phi);
    const double yb = x * std::sin(phi) - y * std::cos(phi);

    if (LRL(xb, yb, phi, t, u, v)) {
        SetPath(paths, {v, u, t}, {'L', 'R', 'L'});
    }
    if (LRL(-xb, yb, -phi, t, u, v)) {
        SetPath(paths, {-v, -u, -t}, {'L', 'R', 'L'});
    }
    if (LRL(xb, -yb, -phi, t, u, v)) {
        SetPath(paths, {v, u, t}, {'R', 'L', 'R'});
    }
    if (LRL(-xb, -yb, phi, t, u, v)) {
        SetPath(paths, {-v, -u, -t}, {'R', 'L', 'R'});
    }
}

std::pair<double, double> CalcTauOmega(double u, double v, double xi, double eta, double phi) {
    const double delta = M(u - v);
    const double A = std::sin(u) - std::sin(delta);
    const double B = std::cos(u) - std::cos(delta) - 1.0;
    const double t1 = std::atan2(eta * A - xi * B, xi * A + eta * B);
    const double t2 = 2.0 * (std::cos(delta) - std::cos(v) - std::cos(u)) + 3.0;

    double tau = 0.0;
    if (t2 < 0) {
        tau = M(t1 + kPi);
    } else {
        tau = M(t1);
    }
    const double omega = M(tau - u + v - phi);
    return {tau, omega};
}

bool LRLRn(double x, double y, double phi, double& t, double& u, double& v) {
    const double xi = x + std::sin(phi);
    const double eta = y - 1.0 - std::cos(phi);
    const double rho = 0.25 * (2.0 + std::hypot(xi, eta));
    if (rho <= 1.0) {
        u = std::acos(rho);
        auto [tau, omega] = CalcTauOmega(u, 0.0, xi, eta, phi);
        if (tau >= 0.0 && omega >= 0.0) {
            t = tau;
            v = omega;
            return true;
        }
    }
    return false;
}

bool LRLRp(double x, double y, double phi, double& t, double& u, double& v) {
    const double xi = x + std::sin(phi);
    const double eta = y - 1.0 - std::cos(phi);
    const double rho = 0.25 * (2.0 + std::hypot(xi, eta));
    if (rho <= 1.0) {
        u = -std::acos(rho);
        auto [tau, omega] = CalcTauOmega(u, 0.0, xi, eta, phi);
        if (tau >= 0.0 && omega >= 0.0) {
            t = tau;
            v = omega;
            return true;
        }
    }
    return false;
}

void CCCC(double x, double y, double phi, std::vector<PathInternal>& paths) {
    double t = 0.0, u = 0.0, v = 0.0;
    if (LRLRn(x, y, phi, t, u, v)) {
        SetPath(paths, {t, u, u, v}, {'L', 'R', 'L', 'R'});
    }
    if (LRLRn(-x, y, -phi, t, u, v)) {
        SetPath(paths, {-t, -u, -u, -v}, {'L', 'R', 'L', 'R'});
    }
    if (LRLRn(x, -y, -phi, t, u, v)) {
        SetPath(paths, {t, u, u, v}, {'R', 'L', 'R', 'L'});
    }
    if (LRLRn(-x, -y, phi, t, u, v)) {
        SetPath(paths, {-t, -u, -u, -v}, {'R', 'L', 'R', 'L'});
    }

    if (LRLRp(x, y, phi, t, u, v)) {
        SetPath(paths, {t, u, u, v}, {'L', 'R', 'L', 'R'});
    }
    if (LRLRp(-x, y, -phi, t, u, v)) {
        SetPath(paths, {-t, -u, -u, -v}, {'L', 'R', 'L', 'R'});
    }
    if (LRLRp(x, -y, -phi, t, u, v)) {
        SetPath(paths, {t, u, u, v}, {'R', 'L', 'R', 'L'});
    }
    if (LRLRp(-x, -y, phi, t, u, v)) {
        SetPath(paths, {-t, -u, -u, -v}, {'R', 'L', 'R', 'L'});
    }
}

bool LRSR(double x, double y, double phi, double& t, double& u, double& v) {
    const double xi = x + std::sin(phi);
    const double eta = y - 1.0 - std::cos(phi);
    auto [rho, theta] = R(xi, eta);
    if (rho >= 2.0) {
        const double r = std::sqrt(rho * rho - 4.0);
        const double u1 = std::atan2(2.0, r);
        t = M(theta + u1);
        u = r;
        v = M(t + kPi - phi);
        if (t >= 0.0 && v >= 0.0) {
            return true;
        }
    }
    return false;
}

bool LRSL(double x, double y, double phi, double& t, double& u, double& v) {
    const double xi = x - std::sin(phi);
    const double eta = y - 1.0 + std::cos(phi);
    auto [rho, theta] = R(xi, eta);
    if (rho >= 2.0) {
        const double r = std::sqrt(rho * rho - 4.0);
        const double u1 = std::atan2(2.0, r);
        t = M(theta + u1);
        u = r;
        v = M(t - phi);
        if (t >= 0.0 && v >= 0.0) {
            return true;
        }
    }
    return false;
}

void CCSC(double x, double y, double phi, std::vector<PathInternal>& paths) {
    double t = 0.0, u = 0.0, v = 0.0;
    if (LRSL(x, y, phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'L', 'R', 'S'});
    }
    if (LRSL(-x, y, -phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'L', 'R', 'S'});
    }
    if (LRSL(x, -y, -phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'R', 'L', 'S'});
    }
    if (LRSL(-x, -y, phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'R', 'L', 'S'});
    }

    if (LRSR(x, y, phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'L', 'R', 'S'});
    }
    if (LRSR(-x, y, -phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'L', 'R', 'S'});
    }
    if (LRSR(x, -y, -phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'R', 'L', 'S'});
    }
    if (LRSR(-x, -y, phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'R', 'L', 'S'});
    }
}

bool LRSLR(double x, double y, double phi, double& t, double& u, double& v) {
    const double xi = x + std::sin(phi);
    const double eta = y - 1.0 - std::cos(phi);
    auto [rho, theta] = R(xi, eta);
    if (rho >= 4.0) {
        const double r = std::sqrt(rho * rho - 4.0);
        double u1 = std::atan2(2.0, r);
        t = M(theta + u1);
        u = M(kPi - 2.0 * u1);
        v = M(phi - t + u);
        if (t >= 0.0 && u >= 0.0 && v >= 0.0) {
            return true;
        }
    }
    return false;
}

void CCSCC(double x, double y, double phi, std::vector<PathInternal>& paths) {
    double t = 0.0, u = 0.0, v = 0.0;
    if (LRSLR(x, y, phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'L', 'R', 'S', 'L', 'R'});
    }
    if (LRSLR(-x, y, -phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'L', 'R', 'S', 'L', 'R'});
    }
    if (LRSLR(x, -y, -phi, t, u, v)) {
        SetPath(paths, {t, u, v}, {'R', 'L', 'S', 'R', 'L'});
    }
    if (LRSLR(-x, -y, phi, t, u, v)) {
        SetPath(paths, {-t, -u, -v}, {'R', 'L', 'S', 'R', 'L'});
    }
}

void GenerateLocalCourse(double L,
                         const std::vector<double>& lengths,
                         const std::vector<char>& mode,
                         double maxc,
                         double step_size,
                         std::vector<double>& px,
                         std::vector<double>& py,
                         std::vector<double>& pyaw,
                         std::vector<int>& directions) {
    const int point_num = static_cast<int>(L / step_size) + static_cast<int>(lengths.size()) + 3;
    px.assign(point_num, 0.0);
    py.assign(point_num, 0.0);
    pyaw.assign(point_num, 0.0);
    directions.assign(point_num, 0);

    int ind = 1;
    directions[0] = lengths[0] > 0.0 ? 1 : -1;
    double d = lengths[0] > 0.0 ? step_size : -step_size;
    double pd = d;
    double ll = 0.0;

    for (size_t i = 0; i < mode.size(); ++i) {
        const char m = mode[i];
        const double l = lengths[i];
        d = l > 0.0 ? step_size : -step_size;

        const double ox = px[ind];
        const double oy = py[ind];
        const double oyaw = pyaw[ind];

        ind -= 1;
        if (i >= 1 && (lengths[i - 1] * lengths[i]) > 0) {
            pd = -d - ll;
        } else {
            pd = d - ll;
        }

        while (std::abs(pd) <= std::abs(l)) {
            ind += 1;
            if (m == 'S') {
                px[ind] = ox + pd / maxc * std::cos(oyaw);
                py[ind] = oy + pd / maxc * std::sin(oyaw);
                pyaw[ind] = oyaw;
            } else {
                const double ldx = std::sin(pd) / maxc;
                double ldy = (1.0 - std::cos(pd)) / maxc;
                if (m == 'R') {
                    ldy = -ldy;
                }
                const double gdx = std::cos(-oyaw) * ldx + std::sin(-oyaw) * ldy;
                const double gdy = -std::sin(-oyaw) * ldx + std::cos(-oyaw) * ldy;
                px[ind] = ox + gdx;
                py[ind] = oy + gdy;
                pyaw[ind] = (m == 'L') ? oyaw + pd : oyaw - pd;
            }
            directions[ind] = pd > 0.0 ? 1 : -1;
            pd += d;
        }

        ll = l - pd - d;
        ind += 1;
        if (m == 'S') {
            px[ind] = ox + l / maxc * std::cos(oyaw);
            py[ind] = oy + l / maxc * std::sin(oyaw);
            pyaw[ind] = oyaw;
        } else {
            const double ldx = std::sin(l) / maxc;
            double ldy = (1.0 - std::cos(l)) / maxc;
            if (m == 'R') {
                ldy = -ldy;
            }
            const double gdx = std::cos(-oyaw) * ldx + std::sin(-oyaw) * ldy;
            const double gdy = -std::sin(-oyaw) * ldx + std::cos(-oyaw) * ldy;
            px[ind] = ox + gdx;
            py[ind] = oy + gdy;
            pyaw[ind] = (m == 'L') ? oyaw + l : oyaw - l;
        }
        directions[ind] = l > 0.0 ? 1 : -1;
    }

    while (!px.empty() && px.back() == 0.0) {
        px.pop_back();
        py.pop_back();
        pyaw.pop_back();
        directions.pop_back();
    }
}

std::vector<PathInternal> GeneratePath(const std::vector<double>& q0,
                                       const std::vector<double>& q1,
                                       double maxc) {
    const double dx = q1[0] - q0[0];
    const double dy = q1[1] - q0[1];
    const double dth = q1[2] - q0[2];
    const double c = std::cos(q0[2]);
    const double s = std::sin(q0[2]);
    const double x = (c * dx + s * dy) * maxc;
    const double y = (-s * dx + c * dy) * maxc;

    std::vector<PathInternal> paths;
    SCS(x, y, dth, paths);
    CSC(x, y, dth, paths);
    CCC(x, y, dth, paths);
    CCCC(x, y, dth, paths);
    CCSC(x, y, dth, paths);
    CCSCC(x, y, dth, paths);

    return paths;
}

}  // namespace

std::vector<ReedsSheppPath> CalcAllPaths(double sx,
                                        double sy,
                                        double syaw,
                                        double gx,
                                        double gy,
                                        double gyaw,
                                        double maxc,
                                        double step_size) {
    const std::vector<double> q0 = {sx, sy, syaw};
    const std::vector<double> q1 = {gx, gy, gyaw};

    auto paths = GeneratePath(q0, q1, maxc);
    std::vector<ReedsSheppPath> out;
    out.reserve(paths.size());

    for (const auto& path : paths) {
        std::vector<double> x, y, yaw;
        std::vector<int> directions;
        GenerateLocalCourse(path.total_length, path.lengths, path.ctypes, maxc, step_size * maxc, x, y, yaw, directions);

        ReedsSheppPath rs;
        rs.lengths.resize(path.lengths.size());
        for (size_t i = 0; i < path.lengths.size(); ++i) {
            rs.lengths[i] = path.lengths[i] / maxc;
        }
        rs.ctypes = path.ctypes;
        rs.total_length = path.total_length / maxc;
        rs.directions = directions;

        rs.x.reserve(x.size());
        rs.y.reserve(y.size());
        rs.yaw.reserve(yaw.size());
        for (size_t i = 0; i < x.size(); ++i) {
            rs.x.push_back(std::cos(-q0[2]) * x[i] + std::sin(-q0[2]) * y[i] + q0[0]);
            rs.y.push_back(-std::sin(-q0[2]) * x[i] + std::cos(-q0[2]) * y[i] + q0[1]);
            rs.yaw.push_back(M(yaw[i] + q0[2]));
        }
        out.push_back(std::move(rs));
    }

    return out;
}

ReedsSheppPath CalcOptimalPath(double sx,
                               double sy,
                               double syaw,
                               double gx,
                               double gy,
                               double gyaw,
                               double maxc,
                               double step_size) {
    auto paths = CalcAllPaths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size);
    if (paths.empty()) {
        return {};
    }
    ReedsSheppPath best = paths.front();
    for (const auto& path : paths) {
        if (path.total_length <= best.total_length) {
            best = path;
        }
    }
    return best;
}

}  // namespace hope
