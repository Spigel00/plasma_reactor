# Phase 2 – Ideation (Multiple Concepts)

## Overview

Phase 2 explored multiple technical approaches to solve the validated computational bottleneck in plasma control systems. The ideation process systematically evaluated different control paradigms, surrogate modeling techniques, and implementation strategies to identify the most promising solution path. This comprehensive exploration considered both traditional control methods and cutting-edge machine learning approaches.

The ideation phase recognized that solving the plasma control challenge required innovation at multiple levels: the control algorithm architecture, the physics approximation method, and the software implementation strategy. Each dimension offered multiple potential solutions that needed evaluation against criteria of speed, accuracy, implementability, and research impact.

## Approach

**Control Paradigm Evaluation:**
We compared classical control approaches (PID controllers, model predictive control) against modern reinforcement learning methods. Classical approaches offered proven stability and interpretability but limited adaptability to complex plasma dynamics. RL approaches promised superior performance in high-dimensional, nonlinear control problems but required the fast simulation capability that motivated the entire project.

**Surrogate Model Architecture Analysis:**
Multiple surrogate modeling approaches were systematically evaluated: analytical plasma approximations (0D/1D models), reduced-order modeling (POD/DMD), physics-informed neural networks, and linear response matrices. Each approach offered different trade-offs between speed, accuracy, and implementation complexity. Linear response matrices emerged as particularly attractive due to their interpretability and sub-millisecond inference capability.

**Implementation Strategy Exploration:**
The technical implementation explored various software frameworks including OpenAI Gym environment development, TORAX integration methods, and surrogate model interfaces. The Gym-based approach provided standardized RL integration while maintaining flexibility for future extensions to different plasma scenarios and control objectives.

## Evidence of Validation

**Expert Feedback on Approach Selection:**
Plasma physics mentors provided critical guidance on the technical approach, highlighting the novelty of applying RL to plasma control and the potential for real-world transferability. Domain experts validated that linear response matrices capture the essential control relationships needed for shape and position control while maintaining computational efficiency.

**Technical Feasibility Analysis:**
Early prototyping confirmed the viability of the chosen approach through successful development of linear surrogate models from TORAX simulation data. Initial testing demonstrated the required 1000x speedup (sub-millisecond inference vs. minutes for full simulation) while maintaining acceptable accuracy for control-relevant observables.

**Literature Validation:**
Comprehensive literature review confirmed the novelty of the RL-based plasma control approach and identified successful applications of similar surrogate modeling techniques in adjacent physics domains. This evidence supported the technical feasibility and potential research impact of the chosen direction.

## Outcome

Phase 2 successfully converged on the optimal technical approach: RL-based control using linear surrogate models implemented through OpenAI Gym environments. This approach uniquely combines the adaptability of reinforcement learning with the speed requirements for practical training, while maintaining the physics interpretability essential for plasma control applications.

**Selected Architecture:**
- **Control Method:** Reinforcement Learning with continuous action spaces for coil current control
- **Surrogate Model:** Linear response matrices mapping control inputs to plasma observables
- **Implementation:** OpenAI Gym environment with TORAX-derived physics data
- **Interface:** Fast Python surrogate model with sub-millisecond inference

**Key Design Decisions:**
1. **RL over Classical Control:** Chosen for superior adaptability to complex, nonlinear plasma dynamics and potential for discovering novel control strategies
2. **Linear Surrogate Models:** Selected for optimal balance of speed, interpretability, and control-relevant accuracy
3. **Gym Environment:** Adopted for standardized RL integration and community accessibility
4. **Modular Architecture:** Designed for extensibility to different plasma scenarios and control objectives

**Validation of Novelty:**
Customer/mentor feedback consistently highlighted the innovative nature of applying RL to plasma control, with particular emphasis on the potential for real-world transferability to experimental plasma devices. The approach was validated as both technically sound and scientifically significant, providing a foundation for breakthrough applications in autonomous fusion reactor control.

This comprehensive ideation process ensured that the selected approach optimally addresses the validated problem while maximizing potential for research impact and practical application.