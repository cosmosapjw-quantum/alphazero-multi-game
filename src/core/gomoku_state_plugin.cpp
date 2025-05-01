// gomoku_state_plugin.cpp
#include "gomoku_state.h"
#include "game_registration.h"

namespace alphazero::gomoku {

REGISTER_GAME(
    gomoku,
    [](const core::VariantArgs& args) -> std::unique_ptr<core::IGameState> {
        int boardSize = args.get<int>("boardSize", 15);
        bool useRenju = args.get<bool>("useRenju", false);
        bool useOmok = args.get<bool>("useOmok", false);
        int seed = args.get<int>("seed", 0);
        bool usePro = args.get<bool>("useProLongOpening", false);
        
        return std::make_unique<GomokuState>(boardSize, useRenju, useOmok, seed, usePro);
    });

} // namespace alphazero::gomoku