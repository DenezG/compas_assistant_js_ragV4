import Image from 'next/image';
import insee_bg from '../datas/insee_bg.png'


export default function Background() {
    return(
        <Image
            alt="Background"
            src={insee_bg}
            placeholder="blur"
            quality={100}
            fill
            sizes="100vw"
            style={{
                objectFit: 'cover',
                zIndex: -5
            }}
        />
        
    );
}