import useInferredFeatures from "@/hooks/useInferredFeatures"
import { Translator } from "@/hooks/useTranslatorAndReplacements"
import { Result } from "@/types"

type HeaderProps = Result & {
  translator: Translator
}

const Header = (props: HeaderProps) => {
  const { config } = props
  const { t } = props.translator  // langIndex, setLangIndex, languages を削除
  const { hasTranslations } = useInferredFeatures(props)

  return (
    <div className='fixed top-0 w-full h-7 bg-gradient-to-r from-blue-900 to-white z-10 leading-7'>
      <div className="flex justify-between">
        <div className="text-white mx-2">
          Talk to the City
        </div>
        {/* 言語切り替えが不要な場合、この部分をコメントアウト */}
        {/* {hasTranslations && (
          <div>
            日本語
          </div>
        )} */}
      </div>
    </div>
  )
}

export default Header
