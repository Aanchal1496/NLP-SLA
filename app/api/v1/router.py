from fastapi import APIRouter

from app.api.v1.analytics import flat_router as analytics_flat_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.analyze import router as analyze_router
from app.api.v1.documents import router as documents_router
from app.api.v1.entities import flat_router as entities_flat_router
from app.api.v1.entities import router as entities_router
from app.api.v1.entities import trained_router as entities_trained_router
from app.api.v1.intent import flat_router as intent_flat_router
from app.api.v1.intent import router as intent_router
from app.api.v1.preprocess import router as preprocess_router
from app.api.v1.translate import router as translate_router

router = APIRouter(prefix="/api/v1")
router.include_router(documents_router, prefix="/documents", tags=["documents"])
router.include_router(preprocess_router, tags=["preprocessing"])
router.include_router(translate_router, tags=["translation"])
router.include_router(intent_router, prefix="/predict", tags=["prediction"])
router.include_router(entities_router, prefix="/predict", tags=["prediction"])
router.include_router(entities_trained_router, prefix="/predict", tags=["prediction"])
router.include_router(entities_trained_router, prefix="/model", tags=["prediction"])
router.include_router(analytics_router, prefix="/analytics", tags=["analytics"])
# Required integration paths (flat aliases + combined analysis + export).
router.include_router(analyze_router, tags=["analysis"])
router.include_router(intent_flat_router, tags=["prediction"])
router.include_router(entities_flat_router, tags=["prediction"])
router.include_router(analytics_flat_router, tags=["analytics"])
